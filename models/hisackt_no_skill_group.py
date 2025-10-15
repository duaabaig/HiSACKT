import os
import numpy as np
import torch
import pickle
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
import statistics
import pandas as pd

class HiSACKT_NoSkillGroup(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout, ha_num_heads_per_level=[2, 2, 2, 2], local_window=25, ckpt_path=None):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.ha_num_heads_per_level = [num_attn_heads] * 4
        self.local_window = local_window

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        kaiming_normal_(self.P)

        self.attn_base = MultiheadAttention(self.d, self.ha_num_heads_per_level[0], dropout=self.dropout)
        self.attn_local = MultiheadAttention(self.d, self.ha_num_heads_per_level[1], dropout=self.dropout)
        self.attn_global = MultiheadAttention(self.d, self.ha_num_heads_per_level[3], dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.attn_cross_base_local = MultiheadAttention(self.d, num_attn_heads, dropout=self.dropout)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)
        self.pred = Linear(self.d, 1)
        self.ckpt_path = ckpt_path

    def forward(self, q, r, qry):
        batch_size, seq_len = q.shape
        q = q[:, :self.n]
        r = r[:, :self.n]
        qry = qry[:, :self.n]
        x = q + self.num_q * r
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(torch.ones([self.n, self.n]), diagonal=1).bool()
        M = M + P

        S_base, attn_weights_base = self.attn_base(E, M, M, attn_mask=causal_mask)
        S_base = self.attn_dropout(S_base)
        S_base = S_base.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)
        S_base = self.attn_layer_norm(S_base + M + E)

        local_seq_len = self.local_window
        num_heads = self.attn_local.num_heads
        local_full_mask = torch.triu(torch.ones([local_seq_len, local_seq_len]), diagonal=1).bool()
        head_mask = local_full_mask.unsqueeze(0).expand(num_heads, -1, -1)
        local_mask = head_mask.unsqueeze(1).expand(num_heads, batch_size, -1, -1)
        local_mask = local_mask.transpose(0, 1).reshape(batch_size * num_heads, local_seq_len, local_seq_len)
        S_local_input = S_base[:, -local_seq_len:, :]
        S_local_input = S_local_input.transpose(0, 1)
        S_local, _ = self.attn_local(S_local_input, S_local_input, S_local_input, attn_mask=local_mask)
        S_local = S_local.transpose(1, 0)
        S_local = torch.cat([torch.zeros(batch_size, self.n - local_seq_len, self.d).to(q.device), S_local], dim=1)

        S_global, _ = self.attn_global(S_base.transpose(0, 1), S_base.transpose(0, 1), S_base.transpose(0, 1), attn_mask=causal_mask)
        S_global = S_global.transpose(0, 1)

        S_base_local, _ = self.attn_cross_base_local(S_base.transpose(0, 1), S_local.transpose(0, 1), S_local.transpose(0, 1))
        S_base_local = S_base_local.transpose(0, 1)

        S = S_base + S_local + S_global + S_base_local
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()
        return p, attn_weights_base

    def train_model(self, train_loader, val_loader, test_loader, num_epochs, opt, ckpt_path, patience=10):
        aucs = []
        loss_means = []
        val_aucs = []
        max_val_auc = 0
        epochs_no_improve = 0

        evaluations = []

        model_output_dir = os.path.join(ckpt_path, "hisackt_no_skill_group")
        os.makedirs(model_output_dir, exist_ok=True)

        for i in range(1, num_epochs + 1):
            loss_mean = []
            for data in train_loader:
                q, r, qshft, rshft, m = data
                self.train()
                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)
                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()
                loss_mean.append(loss.detach().cpu().numpy())

            val_p_all = []
            val_t_all = []
            with torch.no_grad():
                for data in val_loader:
                    q, r, qshft, rshft, m = data
                    self.eval()
                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()
                    val_p_all.append(p)
                    val_t_all.append(t)
                val_p_all = torch.cat(val_p_all)
                val_t_all = torch.cat(val_t_all)
                val_auc = metrics.roc_auc_score(y_true=val_t_all.numpy(), y_score=val_p_all.numpy())
                val_aucs.append(val_auc)

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data
                    self.eval()
                    p, _ = self(q.long(), r.long(), qshft.long())
                    m = m.detach().cpu()
                    p = p.detach().cpu()
                    t = rshft.detach().cpu()

                    p_flat = torch.masked_select(p, m).numpy()
                    t_flat = torch.masked_select(t, m).numpy()
                    auc = metrics.roc_auc_score(y_true=t_flat, y_score=p_flat)
                    aucs.append(auc)

            loss_mean = np.mean(loss_mean)
            loss_means.append(loss_mean)
            evaluations.append({"epoch": i, "auc": auc, "loss-mean": loss_mean})

            if val_auc > max_val_auc:
                torch.save(self.state_dict(), os.path.join(model_output_dir, "model.ckpt"))
                max_val_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        with open(os.path.join(model_output_dir, "val_aucs.pkl"), "wb") as f:
            pickle.dump(val_aucs, f)

        val_aucs_df = pd.DataFrame({
            "epoch": range(1, len(val_aucs) + 1),
            "val_auc": val_aucs
        })
        val_aucs_df.to_csv(os.path.join(model_output_dir, "val_aucs.csv"), index=False)

        evaluations_df = pd.DataFrame(evaluations)
        evaluations_df.to_csv(os.path.join(model_output_dir, "evaluations.csv"), index=False)

        return aucs, loss_means, max_val_auc
