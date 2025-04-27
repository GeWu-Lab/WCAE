import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class Double_Encoder(nn.Module): # two modalities

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Double_Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HAN(nn.Module):

    def __init__(self, num_layers = 1): 
        super(HAN, self).__init__()

        self.fc_a =  nn.Linear(768, 512)
        self.fc_v = nn.Linear(1536, 512)
        self.fc_vst = nn.Linear(1024, 512)
        self.fc_vfusion = nn.Linear(1024, 512)
        self.fc_t =  nn.Linear(768, 512)

        self.fc_out = nn.Linear(1024, 5)  

        self.encoder = Double_Encoder(HANLayer(d_model=512, nhead=8, dim_feedforward=512), num_layers=num_layers)

    def forward(self, audio, visual, video):
    
        # audio
        x1 = self.fc_a(audio)
        # visual
        xv = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        xv = F.avg_pool2d(xv, (8, 1))
        xv = xv.squeeze(-1).permute(0, 2, 1) 
        x_vst = self.fc_vst(video)
        x2 = torch.cat((xv, x_vst), dim =-1) 
        x2 = self.fc_vfusion(x2)

        # HAN
        x1, x2 = self.encoder(x1, x2)
        
        # avg pool in temporal dim
        x1 = torch.mean(x1, dim=1, keepdim=False)
        x2 = torch.mean(x2, dim=1, keepdim=False)
        # prediction
        x = torch.cat([x1, x2], dim=-1)  # [16, 1024]
        
        out = self.fc_out(x)

        return out




