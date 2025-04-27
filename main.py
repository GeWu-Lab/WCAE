
from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ADES_dataset, ToTensor
from nets.net_audiovisual import HAN
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import AverageMeter
import math


def get_batch_qs(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    assert pred.shape[0] == target.shape[0]
    assert pred.shape[1] == target.shape[1]
    idx = np.argmax(pred, axis=1)
    qs = target[np.arange(target.shape[0]), idx]
    batch_qs = np.mean(qs)
    return batch_qs 


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        _, gt = targets.topk(1, 1, largest=True, sorted=True)
        
        pred = pred.t()
        gt = gt.t()
        correct = pred.eq(gt)
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size



def cal_sample_confidence(logit):
    # Reference: An Analysis of Active Learning Strategies for Sequence Labeling Tasks. EMNLP 2008.
    max_logit, _ = torch.max(logit, dim=1)
    max_logit = -max_logit 

    pred = F.softmax(logit, dim=1)
    max_pred, _ = torch.max(pred, dim=1)
    max_pred = 1 - max_pred

    entropy = -torch.sum(pred * pred.log(), dim=1)
    
    max_logit = max_logit.cpu().numpy().tolist()
    max_pred = max_pred.cpu().numpy().tolist()
    entropy = entropy.cpu().numpy().tolist()

    return max_logit, max_pred, entropy


def uncertainty_rank(uncertainty, args):

    rank = [0 for _ in range(len(uncertainty))]
    sorted_index = np.argsort(uncertainty).tolist()
    count = 0
    for i in sorted_index:
        rank[i] = count
        count += 1
    rank = np.array(rank)
    rank = rank / len(rank)

    if args.unc_rank_type == 'equal':
        rank = np.ones(len(rank))
    
    elif args.unc_rank_type == 'sigmoid_sym':
        assert args.sig_sym_th > 0
        rank = rank - 0.5 
        rank = rank * args.sig_sym_th
        rank = 1 / (1 + np.exp(-rank))
    
    else:
        raise ValueError
    
    return rank
    

def train(args, model, train_loader, optimizer, criterion, epoch):
    
    QS = AverageMeter()
    top1_acc = AverageMeter()
    losses = AverageMeter()
    
    model.train()

    for batch_idx, sample in enumerate(train_loader):
        audio, visual, video, qs = sample['audio'].to('cuda'), \
                sample['visual'].to('cuda'), \
                sample['video'].to('cuda'), \
                sample['qs'].to('cuda')

        min_values, _ = torch.min(qs, dim=1) 
        target = (qs == min_values.view(-1, 1)) 
        target = target.int().float() 

        optimizer.zero_grad()
        prob = model(audio, visual, video)

        loss = criterion(prob, target)
        loss = torch.mean(loss)

        batch_acc = calculate_accuracy(prob, target)
        batch_qs = get_batch_qs(prob, qs)

        losses.update(loss.item(),prob.shape[0])
        QS.update(batch_qs, prob.shape[0])
        top1_acc.update(batch_acc,prob.shape[0])
        
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 or batch_idx == len(train_loader)-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}({:.6f})\tqs: {:.6f}({:.6f})\tTop1_acc: {:.6f}({:.6f})'.format(
                epoch, batch_idx, len(train_loader),
                       100. * batch_idx / len(train_loader), loss.item(), losses.avg, batch_qs, QS.avg, batch_acc, top1_acc.avg))

    return QS.avg, top1_acc.avg


def eval(args, model, val_loader, criterion, epoch, phase):
    QS = AverageMeter()
    top1_acc = AverageMeter()
    losses = AverageMeter()
    
    model.eval()

    max_logit = []
    max_pred = []
    entropy = []

    with torch.no_grad():  # get uncertainty
        for batch_idx, sample in enumerate(val_loader):
            audio, visual,video, qs = sample['audio'].to('cuda'), \
                sample['visual'].to('cuda'), \
                sample['video'].to('cuda'), \
                sample['qs'].to('cuda')

            min_values, _ = torch.min(qs, dim=1) 
            target = (qs == min_values.view(-1, 1)) 
            target = target.int().float() 

            prob = model(audio, visual, video)   # [batch_size, 5]
            
            batch_max_logit, batch_max_pred, batch_entropy = cal_sample_confidence(prob)
            max_logit = max_logit + batch_max_logit
            max_pred = max_pred + batch_max_pred
            entropy = entropy + batch_entropy

            if batch_idx % 100 == 0:
                print('cal uncertainty iter:', batch_idx)

    if args.uncertainty == 'max_logit':
        uncertainty = max_logit
    elif args.uncertainty == 'max_pred':
        uncertainty = max_pred
    elif args.uncertainty == 'entropy':
        uncertainty = entropy

    rank = uncertainty_rank(uncertainty, args)
    
    with torch.no_grad(): # inference
        for batch_idx, sample in enumerate(val_loader):
            audio, visual,video, qs = sample['audio'].to('cuda'), \
                sample['visual'].to('cuda'), \
                sample['video'].to('cuda'), \
                sample['qs'].to('cuda')

            batch_num = audio.shape[0]

            min_values, _ = torch.min(qs, dim=1) 
            target = (qs == min_values.view(-1, 1))
            target = target.int().float() 

            prob = model(audio, visual, video)   # [batch_size, 5]

            if args.eval_unc_logit_adj_ratio > 0:
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + batch_num
                batch_rank = rank[start_idx:end_idx]
                adj = torch.zeros((batch_num, 5))
                adj[:, -1] = torch.tensor(batch_rank)
                adj = adj.to('cuda')
                prob = prob + args.eval_unc_logit_adj_ratio * adj

            elif args.eval_unc_logit_adj_ratio == 0:
                pass
            else:
                raise ValueError
        
            loss = criterion(prob, target)  
            loss = torch.mean(loss) 
            
            batch_acc = calculate_accuracy(prob,target)
            batch_qs = get_batch_qs(prob, qs)

            losses.update(loss.item(),prob.shape[0])
            QS.update(batch_qs, prob.shape[0])
            top1_acc.update(batch_acc,prob.shape[0])

            if batch_idx % args.log_interval == 0 or batch_idx == len(val_loader)-1:
                print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}({:.6f})\tqs: {:.6f}({:.6f})\tTop1_acc: {:.6f}({:.6f})'.format(phase, 
                    epoch, batch_idx, len(val_loader),
                        100. * batch_idx / len(val_loader), loss.item(), losses.avg, batch_qs, QS.avg, batch_acc, top1_acc.avg))
    
    return QS.avg, top1_acc.avg


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Learning to Predict Advertisement Expansion Moments in Short-Form Video Platforms. ICMR 2025 full paper')
    parser.add_argument("--audio_dir", type=str, default='path to audio_feat_AST', 
                        help="audio feature dir")
    parser.add_argument("--visual_dir", type=str, default='path to Swin_L_384_feature', 
                        help="visual feature dir of Swin")
    parser.add_argument("--video_dir", type=str, default='path to VideoSwin_B_3D_feature', 
                        help="visual feature dir of VideoSwin")
    parser.add_argument("--label_train", type=str, default="path to label_train.json", 
                        help="train json file")
    parser.add_argument("--label_val", type=str, default="path to label_val.json", 
                        help="val json file")
    parser.add_argument("--label_test", type=str, default="path to label_test.json", 
                        help="test json file")
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2_weight_decay', type=float, default=0.0, metavar='LR',
                        help='L2 weight_decay')
    parser.add_argument('--uncertainty', type=str, default = 'entropy', choices=['entropy', 'max_logit', 'max_pred'],
                        help='select method to calculate uncertainty')
    parser.add_argument('--eval_unc_logit_adj_ratio', type=float, default=5.0,
                        help='strength of uncertainty based logit adjustment')
    parser.add_argument('--unc_rank_type', type=str, default = 'sigmoid_sym', choices=['sigmoid_sym', 'equal'],
                        help='different uncertainty based logit adjustment strategies')
    parser.add_argument("--sig_sym_th", type=float, default=2.0,
                        help='hyperparameter beta in the paper')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    for _ in range(10):
        print('\n')
    dict_args = vars(args)
    for key in dict_args:
        print(key,': ', dict_args[key])


    torch.manual_seed(args.seed)

    print('===ECCV 2020 HAN===')
    model = HAN(num_layers=1).to('cuda')
    

    train_dataset = ADES_dataset(label=args.label_train, audio_dir=args.audio_dir, visual_dir=args.visual_dir, video_dir=args.video_dir, transform = transforms.Compose([
                                            ToTensor()]))
    val_dataset = ADES_dataset(label=args.label_val, audio_dir=args.audio_dir, visual_dir=args.visual_dir,video_dir=args.video_dir, transform = transforms.Compose([
                                            ToTensor()]))
    test_dataset = ADES_dataset(label=args.label_test, audio_dir=args.audio_dir, visual_dir=args.visual_dir,video_dir=args.video_dir, transform = transforms.Compose([
                                            ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory = False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory = False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory = False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    final_train_qs = 999
    val_best_qs = 999
    final_test_qs = 999
    
    final_train_acc = -1
    final_val_acc = -1
    final_test_acc = -1
    for epoch in range(1, args.epochs + 1):
        # train
        print('\n------ train epoch {} ------'.format(epoch))
        train_qs, train_top1_acc = train(args, model, train_loader, optimizer, criterion, epoch=epoch)
        scheduler.step(epoch)
        # val and test
        print('\n------ val epoch {} ------'.format(epoch))
        val_qs, val_top1_acc = eval(args, model, val_loader, criterion, epoch=epoch, phase = 'Val')
        print('\n------ test epoch {} ------'.format(epoch))
        test_qs, test_top1_acc = eval(args, model, test_loader, criterion, epoch=epoch, phase = 'Test')
        
        if val_qs < val_best_qs:   # use val qs to select the best epoch
            final_train_qs = train_qs
            val_best_qs = val_qs
            final_test_qs = test_qs 

            final_train_acc = train_top1_acc
            final_val_acc = val_top1_acc
            final_test_acc = test_top1_acc

        
        print('\n------ epoch {} results ------'.format(epoch))
        print('train_qs:', train_qs*100, '  val_qs:', val_qs*100, '  test_qs:', test_qs*100)
        print('train_acc:', train_top1_acc*100, '  val_acc:', val_top1_acc*100, '  test_acc:', test_top1_acc*100)
        print('\n')
        print('final_train_qs:', final_train_qs*100, '  best_val_qs:', val_best_qs*100, '  final_test_qs:', final_test_qs*100)
        print('final_train_acc:', final_train_acc*100, '  final_val_acc:', final_val_acc*100, '  final_test_acc:', final_test_acc*100)
            
if __name__ == '__main__':
    main()