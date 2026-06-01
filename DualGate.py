# coding=utf-8 fast!!!!!!!!!!!
from typing import Tuple
import torch
import torch.nn as nn   
import torch.nn.functional as F
from torch import Tensor
from layers import create_spectral_features, MLP
import math


# ==================== 对偶特征传播卷积层 ====================
class AIGLogicConvDual(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # ---------- 1. 非门----------
        self.neg_proj = nn.Linear(in_dim, out_dim, bias=True)

        # ---------- 2. 正向与门----------
        self.and_proj = nn.Linear(in_dim, out_dim, bias=True)
        self.and_gate = nn.Linear(in_dim, out_dim, bias=True)

        # ---------- 3. 对偶或门----------
        self.or_proj = nn.Linear(in_dim, out_dim, bias=True)
        self.or_gate = nn.Linear(in_dim*3, out_dim, bias=True)

        # ---------- 4. 自环与归一化 ----------
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        
        self.alpha = nn.Parameter(torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.alpha)

    def forward(self, x, x_dual, pos_edge_index, neg_edge_index):
        num_nodes = x.size(0)
        device = x.device
        out = torch.zeros(num_nodes, self.out_dim, device=device)
        out_dual = torch.zeros(num_nodes, self.out_dim, device=device)

        # ========== 1. 非门  ==========
        if neg_edge_index.size(1) > 0:
            dst_neg = neg_edge_index[1]
            src_neg = neg_edge_index[0]

            # 正向目标 ← 对偶源
            neg_h = self.neg_proj(x_dual[src_neg])
            out.scatter_add_(0, dst_neg.unsqueeze(-1).expand_as(neg_h), neg_h)

            # 对偶目标 ← 正向源 
            neg_h_dual = self.neg_proj(x[src_neg])
            out_dual.scatter_add_(0, dst_neg.unsqueeze(-1).expand_as(neg_h_dual), neg_h_dual)

        # ========== 2. 与门  ==========
        if pos_edge_index.size(1) > 0:
            dst_pos = pos_edge_index[1]
            src_pos = pos_edge_index[0]
            E = pos_edge_index.size(1)
            
            h_all = x[src_pos].view(E // 2, 2, self.in_dim)
            h_all_dual = x_dual[src_pos].view(E // 2, 2, self.in_dim)
            
            h1, h2 = h_all.select(1, 0), h_all.select(1, 1)
            h1_dual, h2_dual = h_all_dual.select(1, 0), h_all_dual.select(1, 1)

            # ----- 正向分支： AND -----
            l1 = self.and_proj(h1)
            l2 = self.and_proj(h2)

            gate = torch.sigmoid(self.and_gate(torch.abs(h1 - h2)))
            and_out = gate * (l1 * l2)

            # ----- 对偶分支：OR -----
            l1_dual = self.or_proj(h1_dual)
            l2_dual = self.or_proj(h2_dual)

            gate_a = torch.sigmoid(self.or_gate(torch.cat([h1_dual, h2_dual, torch.abs(h1_dual - h2_dual)], dim=-1)))
            gate_b = 1.0 - gate_a 
            and_out_dual = gate_a * l1_dual + gate_b * l2_dual

            # ----- 聚合 -----
            dst_pos_unique = dst_pos.view(-1, 2)[:, 0]
            out.scatter_add_(0, dst_pos_unique.unsqueeze(-1).expand_as(and_out), and_out)
            out_dual.scatter_add_(0, dst_pos_unique.unsqueeze(-1).expand_as(and_out_dual), and_out_dual)

        # ========== 3. 自环与归一化 ==========
        out = out + self.alpha * self.self_loop(x)
        out_dual = out_dual + self.alpha * self.self_loop(x_dual)

        out = self.norm(out)
        out_dual = self.norm(out_dual)

        return out, out_dual


# ==================== DualGate 对偶版主模型 ====================
class DualGate(nn.Module):
    def __init__(self, args, node_num: int, device: torch.device, in_dim: int = 3,
                 out_dim: int = 256, layer_num: int = 15, dropout: float = 0.05,
                 lamb: float = 5, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.lamb = lamb
        self.device = device
        hidden_dim = out_dim // 2

        self.pos_edge_index = None
        self.neg_edge_index = None
        self.x = None
        self.x_dual = None

        self.conv1 = AIGLogicConvDual(in_dim, hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(AIGLogicConvDual(hidden_dim, hidden_dim, dropout=dropout))

        self.weight = nn.Linear(hidden_dim * 2, out_dim)
        self.readout_prob = MLP(out_dim, out_dim, 1, num_layer=3, p_drop=dropout,
                                norm_layer='layernorm', act_layer='relu')
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        pos_mask = edge_index_s[:, 2] > 0
        neg_mask = edge_index_s[:, 2] < 0

        self.pos_edge_index = edge_index_s[pos_mask][:, :2].t() if torch.any(pos_mask) else \
            torch.zeros((2, 0), dtype=torch.long, device=self.device)
        self.neg_edge_index = edge_index_s[neg_mask][:, :2].t() if torch.any(neg_mask) else \
            torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if self.pos_edge_index.size(1) > 0:
            _, sort_idx = torch.sort(self.pos_edge_index[1])
            self.pos_edge_index = self.pos_edge_index[:, sort_idx]

        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num, dim=self.in_dim
            ).to(self.device)

        self.x = init_emb
        self.x_dual = -init_emb

    def forward(self, init_emb, edge_index_s) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)

        x, x_dual = self.x, self.x_dual

        x, x_dual = self.conv1(x, x_dual, self.pos_edge_index, self.neg_edge_index)
        x, x_dual = F.relu(x), F.relu(x_dual)

        for conv in self.convs:
            x, x_dual = conv(x, x_dual, self.pos_edge_index, self.neg_edge_index)
            x, x_dual = F.relu(x), F.relu(x_dual)

        z = torch.cat([x, x_dual], dim=-1)
        z = F.relu(self.weight(z))
        prob = torch.sigmoid(self.readout_prob(z))

        return z, prob