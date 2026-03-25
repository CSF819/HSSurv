import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(hazards, S, Y, c, alpha=0., eps=1e-7):
    """
    hazards: [B, K]
    S      : [B, K] (cumprod(1 - hazards))；若 None 则现场计算
    Y      : [B]    (离散时间 bin 索引，0..K-1)
    c      : [B]    (是否删失：1=删失, 0=未删失)
    """
    B = Y.numel()
    Y = Y.view(B, 1).long()
    c = c.view(B, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], dim=1)         # [B, K+1]

    # 未删失样本：log S(y) + log h(y)
    uncensored = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp_min(eps)) +
        torch.log(torch.gather(hazards, 1, Y).clamp_min(eps))
    )
    # 删失样本：log S(y+1)
    censored = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp_min(eps))

    neg_l = censored + uncensored
    loss = (1 - alpha) * neg_l + alpha * uncensored
    return loss.mean()

class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, out, gt, alpha=None):
        total = 0.0
        for haz, s in zip(out['hazards'], out['S']):
            total += nll_loss(haz, s, gt['label'], gt['c'], alpha=self.alpha)
        return total


class CohortLoss():
    """
    组成：
      - intra：四路分量（common, synergy, g_spec, p_spec）与原始单模态（gene, path）的一致性
      - inter：与 cohort bank 的对比（InfoNCE 风格）
    需要 preds 中包含：
      preds['decompose'] = [indiv_know, [geno_indiv, path_indiv]]
        - indiv_know: [B, 4, C]
        - geno_indiv: [B, 1, C]
        - path_indiv: [B, 1, C]
      preds['cohort'] = [patient_bank, c]
        - patient_bank: List[len=n_classes]，每个元素形如 [N_i, 4, C]
        - c: 任意标记（沿用你的语义，可为空）
    """
    def __init__(self, temperature=2.0, eps=1e-8):
        self.tau = temperature
        self.eps = eps

    def __call__(self, out, gt):
        if 'decompose' not in out or 'cohort' not in out:
            return torch.tensor(0., device=gt['label'].device)

        device = gt['label'].device
        indiv, origs = out['decompose']
        cohort, _ = out['cohort']

        # shapes
        # indiv: [B,4,C]; origs=[geno_indiv,path_indiv] 各 [B,1,C]
        B, P, C = indiv.shape
        gene = origs[0]                    # [B,1,C]
        path = origs[1]                    # [B,1,C]
        gene = gene.squeeze(1)             # [B,C]
        path = path.squeeze(1)             # [B,C]

        # ---------- intra：分量-原模态对齐 ----------
        # 计算每个样本的 4x2 余弦相似度矩阵
        indiv_n = F.normalize(indiv, dim=-1)                     # [B,4,C]
        gp = torch.stack([gene, path], dim=1)                    # [B,2,C]
        gp_n = F.normalize(gp, dim=-1)
        sim = torch.einsum('bpc,btc->bpt', indiv_n, gp_n)        # [B,4,2]

        # 你原来的 mask：common/synergy 对 gene & path 都正相关；g_spec 只对 gene；p_spec 只对 path
        mask = torch.tensor([[1,1],[0,0],[1,0],[0,1]], device=device, dtype=sim.dtype)  # [4,2]
        # 同 batch 广播
        mask = mask.unsqueeze(0).expand(B, -1, -1)               # [B,4,2]

        intra = (torch.abs(sim) * (1 - mask) - mask * sim).mean(dim=(1,2)) + 1.0
        intra_loss = intra.mean()

        # ---------- inter：与 cohort bank 的对比 ----------
        labels = gt['label'].long().tolist()
        anchors = indiv.mean(dim=1)                               # [B,C]，把 4 路聚成 anchor
        anchors = F.normalize(anchors, dim=-1)

        pos_sims, neg_sims = [], []
        for i in range(B):
            y = labels[i]
            # 取正/负样本集合
            pos_bank = cohort[y] if y < len(cohort) else None     # [N_pos, 4, C]
            if pos_bank is not None and pos_bank.shape[0] > 0:
                pos_vecs = pos_bank.mean(dim=1)                   # [N_pos, C]
                pos_vecs = F.normalize(pos_vecs, dim=-1)
                sp = torch.matmul(pos_vecs, anchors[i:i+1].T).squeeze(-1)  # [N_pos]
                pos_sims.append(sp)
            # 负样本
            neg_list = []
            for j, bank in enumerate(cohort):
                if j == y or bank.shape[0] == 0: continue
                neg_list.append(bank.mean(dim=1))                 # [N_j, C]
            if len(neg_list) > 0:
                neg_vecs = torch.cat(neg_list, dim=0)            # [N_neg, C]
                neg_vecs = F.normalize(neg_vecs, dim=-1)
                sn = torch.matmul(neg_vecs, anchors[i:i+1].T).squeeze(-1)  # [N_neg]
                neg_sims.append(sn)

        # 聚合成 InfoNCE 风格的概率
        if len(pos_sims) == 0 or len(neg_sims) == 0:
            inter_loss = torch.tensor(0., device=device)
        else:
            pos_cat = torch.cat(pos_sims) / self.tau
            neg_cat = torch.cat(neg_sims) / self.tau
            # 稳定计算：log( exp(pos).mean / (exp(pos).mean + exp(neg).mean) )
            ep = torch.exp(pos_cat).mean()
            en = torch.exp(neg_cat).mean()
            inter_loss = -torch.log((ep + self.eps) / (ep + en + self.eps))

        return intra_loss + inter_loss




loss_dict = {
    'nllsurv': NLLSurvLoss(),
    'cohort': CohortLoss(),}

class Loss_factory(nn.Module):
    """
    args.loss 例子：
      'nllsurv,cohort'
      'nllsurv,cohort,id'
      'nllsurv,cohort,mi_0.5'   # mi 损权重 0.5
    解析规则：
      - 逗号分隔各项
      - 每项可写成 name 或 name_weight
    """
    def __init__(self, args):
        super().__init__()
        loss_item = args.loss.split(',')
        self.loss_collection = {}
        for loss_im in loss_item:
            tags = loss_im.split('_')
            if len(tags) == 2:
                name, w = tags[0], float(tags[1])
            else:
                name, w = tags[0], 1.0
            self.loss_collection[name] = w

    def forward(self, preds, target):
        total = torch.tensor(0., device=target['label'].device)
        ldict = {}
        for name, w in self.loss_collection.items():
            if name not in loss_dict:
                continue
            try:
                val = loss_dict[name](preds, target)
                # 确保是张量、在正确设备
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32, device=total.device)
                val = val * float(w)
            except Exception as e:
                # 出错则跳过该项，同时记录 0，避免训练中断
                val = torch.tensor(0., device=total.device)
            ldict[name] = val.detach()
            total = total + val
        return total, ldict
