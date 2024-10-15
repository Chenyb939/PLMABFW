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

class MultiHeadAttention2(nn.Module):

    def __init__(self, head=1, d_model=128, dropout=0.1):
        super(MultiHeadAttention2, self).__init__()
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

