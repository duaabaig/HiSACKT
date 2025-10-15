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
import matplotlib.pyplot as plt
import seaborn as sns


class HiSACKT(Module):
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
        self.attn_skill_group = MultiheadAttention(self.d, self.ha_num_heads_per_level[2], dropout=self.dropout)
        self.attn_global = MultiheadAttention(self.d, self.ha_num_heads_per_level[3], dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.attn_cross_base_local = MultiheadAttention(self.d, num_attn_heads, dropout=self.dropout)
        self.attn_cross_skill_global = MultiheadAttention(self.d, num_attn_heads, dropout=self.dropout)

        self.skill_cluster_weights = Parameter(torch.Tensor(self.num_q, num_q))
        torch.nn.init.xavier_uniform_(self.skill_cluster_weights)
        self.skill_proj = Linear(self.num_q, self.d)
        self.skill_to_embed = Linear(self.n, self.d)

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
        self.clusters_dir = None

    def _save_skill_cluster_graph(self, stage, model_output_dir):
        weights = self.skill_cluster_weights.detach().cpu().numpy()
        dataset_name = os.path.basename(os.path.dirname(self.ckpt_path)) if self.ckpt_path else os.path.basename(os.path.dirname(model_output_dir))
        original_dataset_name = dataset_name
        heatmap_filename = f"{original_dataset_name}-{stage}-heatmap.png" if original_dataset_name else f"temp-{stage}-heatmap.png"
        clustermap_filename = f"{original_dataset_name}-{stage}-clustermap.png" if original_dataset_name else f"temp-{stage}-clustermap.png"

        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Weight')
        plt.title(f'Skill Cluster Weights for {original_dataset_name} ({stage})')
        plt.xlabel('Skill Index')
        plt.ylabel('Skill Index')
        plt.savefig(os.path.join(model_output_dir, heatmap_filename), dpi=300)
        plt.close()

        weights_normalized = weights - np.mean(weights)
        weights_normalized = weights_normalized / np.max(np.abs(weights_normalized))
        weights_df = pd.DataFrame(weights_normalized)

        plt.figure(figsize=(12, 12))
        sns.set(font_scale=1.2)
        clustermap = sns.clustermap(
            weights_df,
            method='ward',
            metric='euclidean',
            cmap='RdBu_r',
            center=0,
            figsize=(12, 10),
            annot=False,
            dendrogram_ratio=0.15,
            row_cluster=False,
            cbar_pos=(0, 0.2, 0.05, 0.6)
        )
        plt.suptitle(f'Skill Cluster Weights for {original_dataset_name} ({stage})', fontsize=14, y=1.0)
        clustermap.savefig(os.path.join(model_output_dir, clustermap_filename), dpi=300, bbox_inches='tight')
        plt.close()

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
        S_local_input = S_base[:, -local_seq_len:, :].transpose(0, 1)
        S_local, _ = self.attn_local(S_local_input, S_local_input, S_local_input, attn_mask=local_mask)
        S_local = S_local.transpose(1, 0)
        S_local = torch.cat([torch.zeros(batch_size, self.n - local_seq_len, self.d).to(q.device), S_local], dim=1)

        skill_weights = torch.softmax(self.skill_cluster_weights, dim=-1)
        q_onehot = torch.nn.functional.one_hot(q, num_classes=self.num_q).float()
        skill_group_repr = torch.matmul(q_onehot, skill_weights)
        skill_group_repr = self.skill_proj(skill_group_repr)
        S_skill_group = torch.matmul(S_base, skill_group_repr.transpose(-1, -2))
        S_skill_group = self.skill_to_embed(S_skill_group)
        S_skill_group, _ = self.attn_skill_group(S_skill_group.transpose(0, 1), S_skill_group.transpose(0, 1), S_skill_group.transpose(0, 1))
        S_skill_group = S_skill_group.transpose(0, 1)

        S_global, _ = self.attn_global(S_base.transpose(0, 1), S_base.transpose(0, 1), S_base.transpose(0, 1), attn_mask=causal_mask)
        S_global = S_global.transpose(0, 1)

        S_base_local, _ = self.attn_cross_base_local(S_base.transpose(0, 1), S_local.transpose(0, 1), S_local.transpose(0, 1))
        S_base_local = S_base_local.transpose(0, 1)
        S_skill_global, _ = self.attn_cross_skill_global(S_skill_group.transpose(0, 1), S_global.transpose(0, 1), S_global.transpose(0, 1))
        S_skill_global = S_skill_global.transpose(0, 1)

        S = S_base + S_local + S_skill_group + S_global + S_base_local + S_skill_global
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        p = torch.sigmoid(self.pred(F)).squeeze()
        return p, attn_weights_base

    def train_model(self, train_loader, val_loader, test_loader, num_epochs, opt, ckpt_path, patience=10):
        aucs, loss_means, val_aucs = [], [], []
        max_val_auc, epochs_no_improve = 0, 0
        predictions, evaluations, misclassified_samples, high_uncertainty_samples = [], [], [], []

        self.model_output_dir = os.path.join(ckpt_path, "hisackt")
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.clusters_dir = os.path.join(ckpt_path, "clusters")
        os.makedirs(self.clusters_dir, exist_ok=True)
        self._save_skill_cluster_graph("pre_training", self.clusters_dir)

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

            val_p_all, val_t_all = [], []
            with torch.no_grad():
                for data in val_loader:
                    q, r, qshft, rshft, m = data
                    self.eval()
                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()
                    val_p_all.append(p)
                    val_t_all.append(t)
                val_p_all, val_t_all = torch.cat(val_p_all), torch.cat(val_t_all)
                val_auc = metrics.roc_auc_score(y_true=val_t_all.numpy(), y_score=val_p_all.numpy())
                val_aucs.append(val_auc)

            epoch_predictions, epoch_misclassified, epoch_uncertainty = [], [], []
            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data
                    self.eval()
                    p, _ = self(q.long(), r.long(), qshft.long())
                    m, p, t = m.detach().cpu(), p.detach().cpu(), rshft.detach().cpu()
                    for j in range(len(q)):
                        valid_mask = m[j]
                        valid_p, valid_t, valid_qshft = p[j][valid_mask].tolist(), t[j][valid_mask].tolist(), qshft[j][valid_mask].detach().cpu().tolist()
                        sample = {"q": q[j].detach().cpu().tolist(), "r": r[j].detach().cpu().tolist(), "qshft": valid_qshft, "p": valid_p, "t": valid_t}
                        epoch_predictions.append(sample)
                        for pred, true in zip(valid_p, valid_t):
                            if (pred > 0.5) != (true > 0.5):
                                epoch_misclassified.append(sample)
                                break
                            if 0.4 < pred < 0.6:
                                epoch_uncertainty.append(sample)
                                break
                    predictions.append({"epoch": i, "samples": epoch_predictions})
                    misclassified_samples.append({"epoch": i, "samples": epoch_misclassified})
                    high_uncertainty_samples.append({"epoch": i, "samples": epoch_uncertainty})

                    p_flat, t_flat = torch.masked_select(p, m).numpy(), torch.masked_select(t, m).numpy()
                    aucs.append(metrics.roc_auc_score(y_true=t_flat, y_score=p_flat))

            loss_mean = np.mean(loss_mean)
            loss_means.append(loss_mean)
            evaluations.append({"epoch": i, "auc": aucs[-1], "loss-mean": loss_mean})

            if val_auc > max_val_auc:
                torch.save(self.state_dict(), os.path.join(self.model_output_dir, "model.ckpt"))
                max_val_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        self._save_skill_cluster_graph("post_training", self.clusters_dir)

        with open(os.path.join(self.model_output_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(predictions, f)
        with open(os.path.join(self.model_output_dir, "evaluations.pkl"), "wb") as f:
            pickle.dump(evaluations, f)
        with open(os.path.join(self.model_output_dir, "misclassified_samples.pkl"), "wb") as f:
            pickle.dump(misclassified_samples, f)
        with open(os.path.join(self.model_output_dir, "high_uncertainty_samples.pkl"), "wb") as f:
            pickle.dump(high_uncertainty_samples, f)
        with open(os.path.join(self.model_output_dir, "val_aucs.pkl"), "wb") as f:
            pickle.dump(val_aucs, f)

        val_aucs_df = pd.DataFrame({"epoch": range(1, len(val_aucs) + 1), "val_auc": val_aucs})
        val_aucs_df.to_csv(os.path.join(self.model_output_dir, "val_aucs.csv"), index=False)
        evaluations_df = pd.DataFrame(evaluations)
        evaluations_df.to_csv(os.path.join(self.model_output_dir, "evaluations.csv"), index=False)

        predictions_rows, misclassified_rows, uncertainty_rows = [], [], []
        for item in predictions:
            epoch = item["epoch"]
            for sample in item["samples"]:
                predictions_rows.append({"epoch": epoch, "q": str(sample["q"]), "r": str(sample["r"]), "qshft": str(sample["qshft"]), "p": str(sample["p"]), "t": str(sample["t"])})
        predictions_df = pd.DataFrame(predictions_rows)
        predictions_df.to_csv(os.path.join(self.model_output_dir, "predictions.csv"), index=False)

        for item in misclassified_samples:
            epoch = item["epoch"]
            for sample in item["samples"]:
                misclassified_rows.append({"epoch": epoch, "q": str(sample["q"]), "r": str(sample["r"]), "qshft": str(sample["qshft"]), "p": str(sample["p"]), "t": str(sample["t"])})
        misclassified_df = pd.DataFrame(misclassified_rows)
        misclassified_df.to_csv(os.path.join(self.model_output_dir, "misclassified_samples.csv"), index=False)

        for item in high_uncertainty_samples:
            epoch = item["epoch"]
            for sample in item["samples"]:
                uncertainty_rows.append({"epoch": epoch, "q": str(sample["q"]), "r": str(sample["r"]), "qshft": str(sample["qshft"]), "p": str(sample["p"]), "t": str(sample["t"])})
        uncertainty_df = pd.DataFrame(uncertainty_rows)
        uncertainty_df.to_csv(os.path.join(self.model_output_dir, "high_uncertainty_samples.csv"), index=False)

        return aucs, loss_means, max(val_aucs)
