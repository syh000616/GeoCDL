import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import numpy as np
import torch.nn.functional as F

from config import TRAIN_CSV, PROT_DIRS as CONFIG_PROT_DIRS, CHECKPOINT_DIR

# 导入自定义组件
from JointInteractionModel import JointInteractionModel
from joint_dataset import JointInteractionDataset, joint_collate_fn
from GNN import build_brics_vocab, CONFIG as DRUG_CONFIG

# 屏蔽 RDKit 的刷屏警告
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ================= 配置与超参数 =================
TRAIN_CV_CSV = str(TRAIN_CSV)
PROT_DIRS = CONFIG_PROT_DIRS
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
N_FOLDS = 5
LAMBDA_PROT = 0.1  # 蛋白对比损失权重
LAMBDA_DRUG = 0.1  # 药物对比损失权重
LR = 1e-4


# ================= Focal Loss  =================
class RobustFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)  # p=σ(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # p_t = p * y + (1 - p) * (1 - y)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # alpha_t = alpha * y + (1 - alpha) * (1 - y)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss  # mathcal{L}_{Focal} = alpha_t * (1 - p_t)^gamma * BCE(p, y)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ================= 全面指标计算 =================
def calculate_metrics(y_true, y_pred_prob):
    y_pred_label = (np.array(y_pred_prob) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_label)
    try:
        auc_score = roc_auc_score(y_true, y_pred_prob)
        aupr_score = average_precision_score(y_true, y_pred_prob)
    except ValueError:
        auc_score = 0.5
        aupr_score = 0.5
    prec = precision_score(y_true, y_pred_label, zero_division=0)
    rec = recall_score(y_true, y_pred_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)

    return {'acc': acc, 'auc': auc_score, 'aupr': aupr_score, 'prec': prec, 'rec': rec, 'f1': f1}


# ================= 核心训练/验证 Epoch =================
def run_epoch(model, loader, criterion, optimizer, scaler, is_train=True):
    model.train() if is_train else model.eval()

    total_loss, total_focal, total_cl_p, total_cl_d = 0, 0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for drug_batch, ms, mq, ws, wq, p_mask, label in loader:
            # 数据转移至 GPU
            drug_batch = {k: v.to(DEVICE) for k, v in drug_batch.items()}
            ms, mq = ms.to(DEVICE), mq.to(DEVICE)
            ws, wq = ws.to(DEVICE), wq.to(DEVICE)
            p_mask = p_mask.to(DEVICE)
            label = label.to(DEVICE).unsqueeze(1)  # 保持形状为 (B, 1)

            # 混合精度前向传播 (来自代码1)
            with autocast():
                outputs = model(drug_batch, ms, mq, ws, wq, p_mask)
                loss_focal = criterion(outputs['logits'], label)
                loss_cl_p = outputs['cl_loss_prot']
                loss_cl_d = outputs['cl_loss_drug']
                # 联合损失函数
                loss = loss_focal + LAMBDA_PROT * loss_cl_p + LAMBDA_DRUG * loss_cl_d

            # 混合精度反向传播与优化
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防止爆炸
                scaler.step(optimizer)
                scaler.update()

            # 记录日志
            bs = label.size(0)
            total_loss += loss.item() * bs
            total_focal += loss_focal.item() * bs
            total_cl_p += loss_cl_p.item() * bs
            total_cl_d += loss_cl_d.item() * bs

            # 收集预测值用于计算整体指标
            probs = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    n = len(loader.dataset)
    losses = {
        'loss': total_loss / n, 'focal': total_focal / n,
        'cl_p': total_cl_p / n, 'cl_d': total_cl_d / n
    }


    metrics = calculate_metrics(all_labels, all_preds)

    return losses, metrics


# ================= 主函数 =================
def main():
    print(">>> 🚀 开始终极版 5折交叉验证训练...")
    brics_vocab = build_brics_vocab(TRAIN_CV_CSV)
    full_dataset = JointInteractionDataset(TRAIN_CV_CSV, brics_vocab, DRUG_CONFIG, PROT_DIRS)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # 初始化全局指标记录字典 (来自代码2)
    fold_results = {'acc': [], 'auc': [], 'aupr': [], 'prec': [], 'rec': [], 'f1': []}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n" + "=" * 15 + f" Fold {fold + 1} / {N_FOLDS} " + "=" * 15)

        train_loader = DataLoader(Subset(full_dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=joint_collate_fn)
        val_loader = DataLoader(Subset(full_dataset, val_ids), batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=joint_collate_fn)

        model = JointInteractionModel(len(brics_vocab)).to(DEVICE)

        # 使用 AdamW 优化器和混合精度 Scaler
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scaler = GradScaler()
        criterion = RobustFocalLoss()

        best_val_auc = 0.0
        best_metrics = {}
        best_epoch_info = ""

        for epoch in range(EPOCHS):
            # 训练与验证
            t_losses, t_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, is_train=True)
            v_losses, v_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, is_train=False)

            current_info = (
                f"Ep {epoch + 1:02d} | Train Loss: {t_losses['loss']:.3f} | "
                f"Val AUC: {v_metrics['auc']:.4f} | AUPR: {v_metrics['aupr']:.4f} | "
                f"F1: {v_metrics['f1']:.4f} | ACC: {v_metrics['acc']:.4f}"
            )
            print(current_info)

            # 保存当前 Fold 的最佳模型
            if v_metrics['auc'] > best_val_auc:
                best_val_auc = v_metrics['auc']
                best_metrics = v_metrics
                best_epoch_info = current_info
                torch.save(
                    model.state_dict(),
                    str(CHECKPOINT_DIR / f"best_model_fold{fold + 1}.pth")
                )

        # 当前 Fold 结束，记录最佳指标并打印
        print("-" * 50)
        print(f"🏆 Fold {fold + 1} 最佳结果 -> {best_epoch_info}")
        print(f"💾 权重已保存至: best_model_fold{fold + 1}.pth")

        for key in fold_results.keys():
            fold_results[key].append(best_metrics[key])

    # 所有 Fold 结束，打印用于论文表格的统计数据 (来自代码2)
    print("\n" + "🌟" * 25)
    print("🏆 5折交叉验证 最终平均性能 (用于论文表格) 🏆")
    print(f"ROC-AUC:   {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    print(f"PR-AUPR:   {np.mean(fold_results['aupr']):.4f} ± {np.std(fold_results['aupr']):.4f}")
    print(f"Accuracy:  {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}")
    print(f"Precision: {np.mean(fold_results['prec']):.4f} ± {np.std(fold_results['prec']):.4f}")
    print(f"Recall:    {np.mean(fold_results['rec']):.4f} ± {np.std(fold_results['rec']):.4f}")
    print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    print("🌟" * 25 + "\n")


if __name__ == "__main__":
    main()
