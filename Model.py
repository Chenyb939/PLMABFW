import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, head=1, d_model=32, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(2 * d_model, d_model)
        self.linear_key = nn.Linear(2 * d_model, d_model)
        self.linear_value = nn.Linear(2 * d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)

class ESMberty_model(nn.Module):
    def __init__(self):
        super(ESMberty_model, self).__init__()

        # net virus
        self.v_fc1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=10,
                      kernel_size=(1, 1280),
                      stride=(1, 1),
                      padding=(0, 0),
                      ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.v_con = torch.nn.Sequential(
            nn.Conv2d(in_channels=10,
                out_channels=10,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            nn.Conv2d(in_channels=10,
                out_channels=10,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            nn.Conv2d(in_channels=10,
                out_channels=10,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.v_fc2 = torch.nn.Sequential(
            nn.Linear(2500, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU()
        )

        self.emb_fc1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=5,
                      kernel_size=(1, 256),
                      stride=(1, 1),
                      padding=(0, 0),
                      ),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU()
        )

        self.emb_fc2 = torch.nn.Sequential(
            nn.Linear(6400, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        # net heavy
        self.h_fc1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=10,
                      kernel_size=(1, 512),
                      stride=(1, 1),
                      padding=(0, 0),
                      ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )
        self.h_fc2 = torch.nn.Sequential(
            nn.Linear(2560, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        # net light
        self.l_fc1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=10,
                      kernel_size=(1, 512),
                      stride=(1, 1),
                      padding=(0, 0),
                      ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )
        self.l_fc2 = torch.nn.Sequential(
            nn.Linear(2560, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.atte = MultiHeadAttention(head=4, d_model=128)
        self.atte1 = MultiHeadAttention(head=4, d_model=32) 
        self.atte2 = MultiHeadAttention(head=4, d_model=32) 

        # net cat
        self.fc1 = torch.nn.Sequential(
            nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x_virus, x_heavy, x_light):

        x1 = self.v_fc1(x_virus)
        x1 = self.v_con(x1)
        x1 = self.dim_down(x1)
        x1 = self.v_fc2(x1)

        x4 = x_virus.permute(0, 1, 3, 2)
        x4 = self.emb_fc1(x4)
        x4 = self.dim_down(x4)
        x4 = self.emb_fc2(x4)

        y3 = self.atte(x4, x1, x1)
        y3 = y3.squeeze(1)
        y3 = self.fc1(y3)

        x2 = self.h_fc1(x_heavy)
        x2 = self.dim_down(x2)
        x2 = self.h_fc2(x2)

        x3 = self.l_fc1(x_light)
        x3 = self.dim_down(x3)
        x3 = self.l_fc2(x3)

        y1 = self.atte1(y3, x2, x2)
        y2 = self.atte2(y3, x3, x3)

        x = torch.cat((y1, y2), -1)
        x = torch.squeeze(x)
        out = self.fc2(x)
        return out

    def dim_down(self, data):
        dim1 = data.shape[0]
        dim2 = data.shape[1]
        dim3 = data.shape[2]
        return data.view([dim1, dim2 * dim3])
