import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd


class ToTensor(object):

    def __call__(self, sample):
        
        audio = sample['audio']
        visual = sample['visual']
        video = sample['video']
        qs = sample['qs']
        
        return {'audio': torch.from_numpy(audio), 'visual': torch.from_numpy(visual), 
                'video': torch.from_numpy(video), 'qs': torch.from_numpy(qs)}


class ADES_dataset(Dataset):

    def __init__(self, label, audio_dir, visual_dir,video_dir, transform=None):
        
        with open(label, encoding='utf-8') as f:
            self.label_dict = json.load(f)
        
        self.name_list = list(self.label_dict.keys())

        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))

        visual = np.load(os.path.join(self.visual_dir, name + '.npy'))

        video = np.load(os.path.join(self.video_dir, name + '.npy'))

        assert audio.shape[0] == 7
        assert visual.shape[0] == 56
        assert video.shape[0] == 7

        all_exp = self.label_dict[name]['all_exp_count']
        qs_exp = self.label_dict[name]['qs_exp_count']
        pre_exp = self.label_dict[name]['pre_exp_count']

        qs = [0 for _ in range(5)]
        for j in range(5):
            qs[j] = qs_exp[j] / (all_exp[j] - pre_exp[j])
        qs = np.array(qs)

        sample = {'audio': audio, 'visual': visual, 'video': video, 'qs': qs}

        if self.transform:
            sample = self.transform(sample)

        return sample
