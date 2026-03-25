# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_SIZES = {
    'brca': {'train': 538, 'val': 128, 'total': 666, 'scale': 'large'},
    'luad': {'train': 353, 'val': 88, 'total': 441, 'scale': 'large'},
    'ucec': {'train': 308, 'val': 76, 'total': 384, 'scale': 'medium'},
    'gbmlgg': {'train': 250, 'val': 62, 'total': 312, 'scale': 'small'},
    'blca': {'train': 218, 'val': 54, 'total': 272, 'scale': 'small'},
}

SCALE_CONFIGS = {
    'large': {
        'path_num_experts': 5,
        'path_topk': 1,
        'expert_dropout': 0.4,
        'gate_temperature': 1.0,
        'gate_noise_std': 0.1,
        'token_keep_ratio': 0.9,
        'bank_length': 20,
        'path_load_balance_weight': 0.2,
        'i2moe_w_con': 0.2,
        'i2moe_w_uni': 0.2,
        'i2moe_w_syn': 0.2,
        'i2moe_w_red': 0.2,
        'i2moe_w_attn': 0.15,
        'i2moe_tau': 0.15,
        'decorr_w': 0.15,
        'attn_temp': 1.0,
    },
    'medium': {
        'path_num_experts': 4,
        'path_topk': 1,
        'expert_dropout': 0.55,
        'gate_temperature': 1.6,
        'gate_noise_std': 0.35,
        'token_keep_ratio': 0.85,
        'bank_length': 20,
        'path_load_balance_weight': 0.3,
        'i2moe_w_con': 0.2,
        'i2moe_w_uni': 0.1,
        'i2moe_w_syn': 0.1,
        'i2moe_w_red': 0.1,
        'i2moe_w_attn': 0.05,
        'i2moe_tau': 0.3,
        'decorr_w': 0.2,
        'attn_temp': 1.3,
    },
    'small': {

        'path_num_experts': 3,
        'path_topk': 1,
        'expert_dropout': 0.7,
        'gate_temperature': 2.0,
        'gate_noise_std': 0.5,
        'token_keep_ratio': 0.7,
        'bank_length': 8,
        'path_load_balance_weight': 0.5,
        'i2moe_w_con': 0.1,
        'i2moe_w_uni': 0.05,
        'i2moe_w_syn': 0.05,
        'i2moe_w_red': 0.05,
        'i2moe_w_attn': 0.03,
        'i2moe_tau': 0.4,
        'decorr_w': 0.25,
        'attn_temp': 1.5,
    }
}


DATASET_SPECIFIC = {
    'brca': {

        'lr': 5e-4,
        'weight_decay': 1e-4,
        'num_epoch': 30,
    },
    'luad': {

        'lr': 2e-4,
        'weight_decay': 1e-5,
        'num_epoch': 35,
    },

    'ucec':{
        'lr': 2e-4,
        'weight_decay': 1e-5,
        'num_epoch': 45,
        'kd_gate_init': [0.7, 0.7, 0.2, 0.2],

        'adaptive_sampling': False,     
        'token_keep_ratio': 1.0,       
        'gate_noise_std': 0.0,          

        'path_num_experts': 1,         
        'path_topk': 1,                 
        'expert_dropout': 0.4,          
        'path_load_balance_weight': 0.0,
        'gate_temperature': 1.2,        

        'i2moe_w_attn': 0.0,            
        'i2moe_w_uni': 0.0,             
        'i2moe_w_syn': 0.0,            
        'i2moe_w_red': 0.0,             
        'i2moe_w_con': 0.0,             
      
        'decorr_w': 0.1,                
        'attn_temp': 1.2,              
        'bank_length': 12,
    },
    'gbmlgg': {
        
        'lr': 1e-4,
        'weight_decay': 5e-6,
        'num_epoch': 30,
        'path_gate_geno_ratio': 0.08,  
    },
    'blca': {
        
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'num_epoch': 50,
        'kd_gate_init': [0.7, 0.7, 0.2, 0.2],  
    }
}


def get_auto_config(dataset_name=None):
    base_config = {
        'modal': 'multi',
        'loss': 'nllsurv,cohort',
        'sets': 'blca,brca,luad,ucec,gbmlgg',
        'lr': 2e-4,
        'optimizer': 'SGD',
        'scheduler': 'cosine',
        'num_epoch': 30,
        'seed': 0,
        'weight_decay': 1e-5,
    }

    customized_config = {
        'spfusion_dim': {'type': int, 'default': 768},
        'max_tokens': {'type': int, 'default': 4096},
        'min_tokens': {'type': int, 'default': 50},
        'adaptive_sampling': {'type': bool, 'default': True},

        'path_num_experts': {'type': int, 'default': 4},
        'path_topk': {'type': int, 'default': 1},
        'path_gate_geno_ratio': {'type': float, 'default': 0.05},
        'path_load_balance_weight': {'type': float, 'default': 0.3},
        'path_lb_hard_usage': {'type': bool, 'default': True},

        'i2moe_w_uni': {'type': float, 'default': 0.1},
        'i2moe_w_syn': {'type': float, 'default': 0.1},
        'i2moe_w_red': {'type': float, 'default': 0.1},
        'i2moe_w_attn': {'type': float, 'default': 0.05},
        'i2moe_w_con': {'type': float, 'default': 0.2},
        'i2moe_tau': {'type': float, 'default': 0.3},

        'bank_length': {'type': int, 'default': 25},

        'align_w': {'type': float, 'default': 1.0},
        'decorr_w': {'type': float, 'default': 0.2},
        'var_w': {'type': float, 'default': 0.0},
        'align_type': {'type': str, 'default': 'cos'},
        'kd_gate_init': {'type': object, 'default': [0.6, 0.6, 0.3, 0.3]},

        'expert_dropout': {'type': float, 'default': 0.6},
        'gate_temperature': {'type': float, 'default': 1.5},
        'gate_noise_std': {'type': float, 'default': 0.3},
        'token_keep_ratio': {'type': float, 'default': 0.8},
        'attn_temp': {'type': float, 'default': 1.3},
    }

    if dataset_name:
        dataset_name = dataset_name.lower()

        if dataset_name in DATASET_SIZES:
            scale = DATASET_SIZES[dataset_name]['scale']
            scale_params = SCALE_CONFIGS[scale]

            for key, value in scale_params.items():
                if key in customized_config:
                    customized_config[key]['default'] = value

            if dataset_name in DATASET_SPECIFIC:
                specific_params = DATASET_SPECIFIC[dataset_name]

                for key in ['lr', 'optimizer', 'num_epoch', 'weight_decay']:
                    if key in specific_params:
                        base_config[key] = specific_params[key]

                for key, value in specific_params.items():
                    if key in customized_config:
                        customized_config[key]['default'] = value

            print(f"Auto-configured for {dataset_name.upper()} (scale: {scale})")
            print(f"  - Experts: {customized_config['path_num_experts']['default']}")
            print(f"  - Top-k: {customized_config['path_topk']['default']}")
            print(f"  - Dropout: {customized_config['expert_dropout']['default']}")
            print(f"  - LR: {base_config['lr']}")
            print(f"  - Optimizer: {base_config['optimizer']}")

    return {
        'base': base_config,
        'customized': customized_config
    }


custom_config = get_auto_config()

# ================= 基因分段索引 =================
genomics_idx = {
    'blca': [0, 94, 428, 949, 1417, 2913, 3392],
    'brca': [0, 91, 444, 997, 1477, 3043, 3523],
    'gbmlgg': [0, 84, 398, 896, 1311, 2707, 3135],
    'luad': [0, 89, 423, 957, 1428, 2938, 3419],
    'ucec': [0, 3, 27, 48, 70, 135, 150],
}


def SNN_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.SELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )

def MLP_Block(dim1, dim2):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.LayerNorm(dim2),
        nn.ReLU()
    )

def conv1d_Block(dim1, dim2):
    return nn.Sequential(
        nn.Conv1d(dim1, dim2, 1),
        nn.InstanceNorm1d(dim2),
        nn.ReLU()
    )

def safe_std(x, dim=None, keepdim=False, eps=1e-12):
    """
    使用总体标准差（unbiased=False），并做数值下限钳位，避免 DoF 警告/NaN
    """
    if dim is None:
        s = x.float().std(unbiased=False)
    else:
        s = x.float().std(dim=dim, keepdim=keepdim, unbiased=False)
    return s.clamp_min(eps)


def random_like_with_stats(x):
    if x.numel() == 0:
        return torch.zeros_like(x)
    mu = x.mean(dim=list(range(x.dim() - 1)), keepdim=True)
    sigma = safe_std(x, dim=list(range(x.dim() - 1)), keepdim=True, eps=1e-6)
    return mu + sigma * torch.randn_like(x)


class Specificity_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=256):
        super().__init__()
        self.conv = MLP_Block(dim, dim)
    def forward(self, feat):
        return self.conv(feat)

class BasicSeparation(nn.Module):
    def __init__(self, dim=256, w_align=1.0, w_decorr=0.1, w_var=0.0, align_type='cos'):
        super().__init__()
        self.dim = dim
        self.w_align, self.w_decorr, self.w_var = w_align, w_decorr, w_var
        self.align_type = align_type
        self.common_enc = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.synergy_enc = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.res_proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _cos_loss(a, b, eps=1e-8):
        a = F.normalize(a, p=2, dim=-1); b = F.normalize(b, p=2, dim=-1)
        return 1.0 - (a * b).sum(dim=-1).mean()
    @staticmethod
    def _decorrelation(a, b, eps=1e-8):
        a = (a - a.mean(0, True)) / (safe_std(a, dim=0, keepdim=True, eps=eps))
        b = (b - b.mean(0, True)) / (safe_std(b, dim=0, keepdim=True, eps=eps))
        return (a * b).mean(0).pow(2).mean()
    @staticmethod
    def _variance_loss(z, target=1.0):
        v = z.var(dim=0); return F.relu(target - v).mean()

    def forward(self, gfeat, pfeat, return_extras=True):
        g = gfeat.squeeze(1); p = pfeat.squeeze(1)
        common = self.common_enc(torch.cat([g, p], dim=-1))
        inter  = torch.cat([g, p, g * p, torch.abs(g - p)], dim=-1)
        synergy_raw = self.synergy_enc(inter)
        synergy = F.layer_norm(synergy_raw - self.res_proj(common.detach()), (self.dim,))
        with torch.no_grad(): g_sg, p_sg = g.clone(), p.clone()
        if self.align_type == 'cos':
            loss_align = 0.5 * (self._cos_loss(common, g_sg) + self._cos_loss(common, p_sg))
        else:
            loss_align = 0.5 * (F.mse_loss(common, g_sg) + F.mse_loss(common, p_sg))
        loss_decorr = self._decorrelation(common, synergy)
        aux_loss = self.w_align * loss_align + self.w_decorr * loss_decorr
        extras = {'mi_common': (-loss_align).detach(),
                  'mi_synergy': (-loss_decorr).detach(),
                  'decorr': loss_decorr.detach()}
        return common.unsqueeze(1), synergy.unsqueeze(1), aux_loss, (extras if return_extras else {})

class Knowledge_Decomposition(nn.Module):
    def __init__(self, feat_len=6, feat_dim=256, align_w=1.0, decorr_w=0.1, var_w=0.0, align_type='cos', kd_gate_init=0.5):
        super().__init__()
        self.geno_spec = Specificity_Estimator(feat_len, feat_dim)
        self.path_spec = Specificity_Estimator(feat_len, feat_dim)
        self.interaction_encoder = BasicSeparation(dim=feat_dim, w_align=align_w, w_decorr=decorr_w, w_var=var_w, align_type=align_type)

        init_val = kd_gate_init
        if isinstance(init_val, (list, tuple)):
            assert len(init_val) == 4
            init_tensor = torch.tensor(init_val, dtype=torch.float32)
        else:
            init_tensor = torch.full((4,), float(init_val))
        self.gates = nn.Parameter(init_tensor)

    def forward(self, gfeat, pfeat, return_losses=True):
        g_spec = self.geno_spec(gfeat)
        p_spec = self.path_spec(pfeat)
        common, synergy, aux_loss, extras = self.interaction_encoder(gfeat, pfeat, return_extras=True)
        parts = torch.cat([common, synergy, g_spec, p_spec], dim=1)
        parts = parts * self.gates.view(1, 4, 1)
        return (parts, (aux_loss, extras)) if return_losses else parts


class TokenI2MoE(nn.Module):
    def __init__(self, in_dim, num_experts=6, top_k=2, use_geno_in_gate=True, gate_geno_ratio=0.1,
                 load_balance_weight=0.01, epsilon=1e-6, lb_hard_usage=True, expert_dropout=0.5,
                 gate_temperature=1.0, gate_noise_std=0.0):
        super().__init__()
        self.C = in_dim
        self.K = int(num_experts)
        self.top_k = max(1, min(top_k, self.K))
        self.use_geno_in_gate = use_geno_in_gate
        self.gate_geno_ratio = gate_geno_ratio
        self.load_balance_weight = load_balance_weight
        self.epsilon = epsilon
        self.lb_hard_usage = lb_hard_usage
        self.gate_temperature = float(gate_temperature)
        self.gate_noise_std   = float(gate_noise_std)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.C, self.C),
                nn.ReLU(inplace=True),
                nn.Dropout(p=expert_dropout),
                nn.Linear(self.C, self.C),
                nn.Dropout(p=expert_dropout)
            )
            for _ in range(self.K)
        ])
        self.gate_proj = nn.Linear(self.C, self.K)
        self.gate_geno = nn.Linear(self.C, self.K) if use_geno_in_gate else None

        for e in self.experts:
            for m in e.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=1e-3); nn.init.zeros_(self.gate_proj.bias)
        if self.gate_geno is not None:
            nn.init.zeros_(self.gate_geno.weight); nn.init.zeros_(self.gate_geno.bias)


    @staticmethod
    def _cv2(x, eps=1e-6):
        m = x.mean()
        s = x.std(unbiased=False)  
        return (s / (m + eps)) ** 2

    def _load_balance_loss(self, gate_logits, topi=None):
        B, N, K = gate_logits.shape
        if self.lb_hard_usage and (topi is not None):
            hit = torch.zeros(B, N, K, dtype=torch.float32, device=gate_logits.device)
            hit.scatter_(2, topi, 1.0)
            usage = hit.mean(dim=(0, 1))
            return self._cv2(usage, self.epsilon)
        else:
            probs = F.softmax(gate_logits, dim=-1)
            usage = probs.mean(dim=(0, 1))
            return self._cv2(usage, self.epsilon)

    def forward(self, tokens, geno_vec=None, keep_topk_snapshot=False):
        """
        tokens: [B,N,C]
        返回: centers[B,K,C], viz_dict, lb_loss
        """
        assert tokens.dim() == 3
        B, N, C = tokens.shape
        device = tokens.device

        if self.use_geno_in_gate and geno_vec is not None:
            if geno_vec.dim() == 3:
                geno_vec = geno_vec.squeeze(1)  # [B,C]

        logits = self.gate_proj(tokens)  # [B,N,K]
        if self.use_geno_in_gate and geno_vec is not None:
            logits = logits + self.gate_geno_ratio * self.gate_geno(geno_vec).unsqueeze(1)

        if self.training and self.gate_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise_std
        logits = logits / max(self.gate_temperature, 1e-6)

        topv, topi = torch.topk(logits, k=self.top_k, dim=-1)  # [B,N,topk]
        topw = F.softmax(topv, dim=-1).clamp_min(self.epsilon)  # [B,N,topk]
        topw = topw / topw.sum(dim=-1, keepdim=True)

        lb_loss = self._load_balance_loss(logits.detach(), topi.detach())

        W = torch.zeros(B, N, self.K, device=device, dtype=tokens.dtype)
        W.scatter_(2, topi, topw)

        mass_sum = W.sum(dim=1, keepdim=False)  # [B,K]
        counts = (W > 0).sum(dim=1)  # [B,K]

        centers_sum = torch.zeros(B, self.K, C, device=device, dtype=tokens.dtype)
        b_idx = torch.arange(B, device=device)[:, None].expand(B, N)
        for k in range(self.K):
            wk = W[..., k]  # [B,N]
            mask = wk > 0  # [B,N]
            num = int(mask.sum().item())
            if num == 0:
                continue
            b_pick = b_idx[mask]  # [M]
            n_pick = torch.nonzero(mask, as_tuple=False)[:, 1]  # [M]
            tok_pick = tokens[b_pick, n_pick, :]  # [M,C]
            w_pick = wk[mask].unsqueeze(-1)  # [M,1]
            out_k = self.experts[k](tok_pick) * w_pick  # [M,C]
            flat_idx = b_pick * self.K + k  # [M]
            centers_sum = centers_sum.view(B * self.K, C)
            centers_sum.index_add_(0, flat_idx, out_k)
            centers_sum = centers_sum.view(B, self.K, C)

        centers = centers_sum / mass_sum.clamp_min(self.epsilon).unsqueeze(-1)  # [B,K,C]

        expert_usage = (counts > 0).float().mean(dim=0)  # [K]
        avg_tokens_per_expert = counts.float().mean(dim=0)  # [K]
        if self.top_k > 1:
            routing_confidence = (topv[:, :, 0] - topv[:, :, 1]).mean()
        else:
            routing_confidence = torch.tensor(0.0, device=logits.device)

        viz = {
            'gate_mass': mass_sum,
            'counts': counts,
            'expert_usage': expert_usage,
            'avg_tokens_per_expert': avg_tokens_per_expert,
            'routing_confidence': routing_confidence,
        }
        if keep_topk_snapshot:
            keepN = min(N, 512)
            viz['topk_idx'] = topi[:, :keepN].detach().cpu()

        return centers, viz, lb_loss


class HSSurv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seed = args.seed
        self.n_classes = args.n_classes
        self.bank_length = args.bank_length
        self.feat_dim = 256
        self.C_sp = getattr(args, 'spfusion_dim', 768)

        self.max_tokens = getattr(args, 'max_tokens', 4096)
        self.min_tokens = getattr(args, 'min_tokens', 50)
        self.adaptive_sampling = getattr(args, 'adaptive_sampling', True)

        self.token_keep_ratio = float(getattr(args, 'token_keep_ratio', 1.0))
        self.gate_temperature = float(getattr(args, 'gate_temperature', 1.0))
        self.gate_noise_std   = float(getattr(args, 'gate_noise_std', 0.0))
        self.attn_temp        = float(getattr(args, 'attn_temp', 1.0))
        self.expert_dropout   = float(getattr(args, 'expert_dropout', 0.5))

        self.i2moe_w_uni = getattr(args, 'i2moe_w_uni', 0.05)
        self.i2moe_w_syn = getattr(args, 'i2moe_w_syn', 0.05)
        self.i2moe_w_red = getattr(args, 'i2moe_w_red', 0.05)
        self.i2moe_w_attn = getattr(args, 'i2moe_w_attn', 0.05)
        self.i2moe_w_con = getattr(args, 'i2moe_w_con', 0.05)
        self.i2moe_tau = getattr(args, 'i2moe_tau', 0.07)
        self.path_lb_weight = getattr(args, 'path_load_balance_weight', 0.01)

        self.path_token_proj = SNN_Block(self.C_sp, self.feat_dim)
        self._path_num_experts = getattr(args, 'path_num_experts', 6)
        self.path_moe = TokenI2MoE(
            in_dim=self.feat_dim,
            num_experts=self._path_num_experts,
            top_k=min(getattr(args, 'path_topk', 2), self._path_num_experts),
            use_geno_in_gate=True,
            gate_geno_ratio=getattr(args, 'path_gate_geno_ratio', 0.1),
            load_balance_weight=self.path_lb_weight,
            lb_hard_usage=getattr(args, 'path_lb_hard_usage', True),
            expert_dropout=self.expert_dropout,          
            gate_temperature=self.gate_temperature,      
            gate_noise_std=self.gate_noise_std,          
        )
        self.path_fc = SNN_Block(self.feat_dim, self.feat_dim)
        self.path_conv = conv1d_Block(self._path_num_experts, 1)

        self.genomics_idx = genomics_idx[args.dataset]
        sig_networks = []
        for idx in range(len(self.genomics_idx) - 1):
            d = self.genomics_idx[idx + 1] - self.genomics_idx[idx]
            sig_networks.append(nn.Sequential(SNN_Block(d, 256)))
        self.genomics_fc = nn.ModuleList(sig_networks)
        self.geno_fc = SNN_Block(256, self.feat_dim)
        self.geno_conv = conv1d_Block(6, 1)

        self.know_decompose = Knowledge_Decomposition(
            feat_len=len(self.genomics_idx) - 1,
            feat_dim=self.feat_dim,
            align_w=getattr(args, 'align_w', 1.0),
            decorr_w=getattr(args, 'decorr_w', 0.1),
            var_w=getattr(args, 'var_w', 0.0),
            align_type=getattr(args, 'align_type', 'cos'),
            kd_gate_init=getattr(args, 'kd_gate_init', 0.5),
        )

        self.num_interactions = 4
        self.expert_queries = nn.Parameter(torch.randn(self.num_interactions, self.feat_dim))
        nn.init.normal_(self.expert_queries, mean=0.0, std=0.02)
        self.expert_residual_gate = nn.Parameter(torch.full((self.num_interactions,), 0.2))
        self.expert_prior_strength = nn.Parameter(torch.full((self.num_interactions,), 2.0))

        self.expert_heads = nn.ModuleList([nn.Linear(self.feat_dim, self.n_classes) for _ in range(self.num_interactions)])
        for head in self.expert_heads:
            nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)

        self.reweight_norm = nn.LayerNorm(self.feat_dim)
        self.reweight_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim // 2, self.num_interactions)
        )
        nn.init.xavier_uniform_(self.reweight_mlp[0].weight); nn.init.zeros_(self.reweight_mlp[0].bias)
        nn.init.xavier_uniform_(self.reweight_mlp[2].weight); nn.init.zeros_(self.reweight_mlp[2].bias)

        # memory banks
        for i in range(self.n_classes):
            self.register_buffer(f"expert_bank_{i}", torch.empty(0, self.num_interactions, self.feat_dim), persistent=True)

    # ===== 工具 =====
    def _adaptive_token_sampling(self, x_path):
        B, N, C = x_path.shape
        if N == 0:
            x_path = torch.randn(B, self.min_tokens, C, device=x_path.device) * 1e-4
            return x_path, self.min_tokens, True
        if not self.adaptive_sampling:
            return x_path, N, False
        sampled = False
        if N > self.max_tokens:
            idx = torch.randperm(N, device=x_path.device)[:self.max_tokens]
            x_path = x_path[:, idx, :]; sampled, N = True, self.max_tokens
        elif N < self.min_tokens:
            repeat_times = math.ceil(self.min_tokens / N)
            x_rep = x_path.repeat(1, repeat_times, 1)[:, :self.min_tokens, :]
            noise = torch.randn_like(x_rep) * 1e-4
            x_path = x_rep + noise; sampled, N = True, self.min_tokens
        return x_path, N, sampled

    @staticmethod
    def _anchor_map():
        return [2, 3, 1, 0]

    def _expert_pool(self, tokens4, return_align_loss=False):
        B, T, C = tokens4.shape
        q = self.expert_queries / math.sqrt(C)
        attn_logits = torch.einsum('ic,btc->bit', q, tokens4)

        anchor_idx = self._anchor_map()
        prior = torch.zeros_like(attn_logits)
        for i, ai in enumerate(anchor_idx): prior[:, i, ai] = self.expert_prior_strength[i]

        attn_logits = (attn_logits + prior) / max(getattr(self, 'attn_temp', 1.0), 1e-6)

        attn  = F.softmax(attn_logits, dim=-1)
        mixed = torch.einsum('bit,btc->bic', attn, tokens4)
        gate = torch.sigmoid(self.expert_residual_gate).view(1, -1, 1)
        anchor_tokens = torch.stack([tokens4[:, ai, :] for ai in anchor_idx], dim=1)
        expert_feats = (1.0 - gate) * anchor_tokens + gate * mixed
        align_loss = torch.tensor(0.0, device=tokens4.device)
        if return_align_loss:
            eps = 1e-8
            anchor_probs = torch.stack([attn[:, i, ai] for i, ai in enumerate(anchor_idx)], dim=1)
            align_loss = (-torch.log(anchor_probs.clamp_min(eps))).mean()
        return expert_feats, attn, align_loss

    @staticmethod
    def _triplet(a, p, n, margin=1.0):
        return nn.TripletMarginLoss(margin=margin, p=2, eps=1e-7)(a, p, n)
    @staticmethod
    def _cosine(a, b):
        a = F.normalize(a, p=2, dim=-1); b = F.normalize(b, p=2, dim=-1)
        return (a * b).sum(dim=-1)

    def _interaction_losses(self, feats_full, feats_gabl, feats_pabl):
        losses = {}
        a = feats_full[:, 0, :]; n = feats_gabl[:, 0, :]; p = feats_pabl[:, 0, :]
        losses['i2moe_uni_g'] = self._triplet(a, p, n)
        a = feats_full[:, 1, :]; n = feats_pabl[:, 1, :]; p = feats_gabl[:, 1, :]
        losses['i2moe_uni_p'] = self._triplet(a, p, n)
        a = feats_full[:, 2, :]; n1 = feats_gabl[:, 2, :]; n2 = feats_pabl[:, 2, :]
        losses['i2moe_synergy'] = (self._cosine(a, n1).mean() + self._cosine(a, n2).mean()) / 2.0
        a = feats_full[:, 3, :]; p1 = feats_gabl[:, 3, :]; p2 = feats_pabl[:, 3, :]
        losses['i2moe_redundancy'] = ((1.0 - self._cosine(a, p1)).mean() + (1.0 - self._cosine(a, p2)).mean()) / 2.0
        return losses

    def _supcon_with_bank(self, expert_feats_full, label):
        device = expert_feats_full.device
        if label is None:
            return torch.tensor(0.0, device=device)
        if isinstance(label, int):
            label = torch.tensor([label], device=device)
        elif torch.is_tensor(label):
            label = label.view(-1).long().to(device)

        B, _, C = expert_feats_full.shape
        feats = F.normalize(expert_feats_full.reshape(B, -1), p=2, dim=-1)
        total_loss, valid = 0.0, 0
        for b in range(B):
            lb = int(label[b].item())
            pos_bank = getattr(self, f"expert_bank_{lb}")
            if pos_bank.shape[0] == 0: continue
            neg_list = [getattr(self, f"expert_bank_{i}") for i in range(self.n_classes) if i != lb and getattr(self, f"expert_bank_{i}").shape[0] > 0]
            if len(neg_list) == 0: continue
            neg_bank = torch.cat(neg_list, dim=0)
            pos = F.normalize(pos_bank.reshape(-1, 4 * C), p=2, dim=-1)
            neg = F.normalize(neg_bank.reshape(-1, 4 * C), p=2, dim=-1)
            tau = self.i2moe_tau
            feat_b = feats[b:b+1]
            logits_pos = feat_b @ pos.t() / tau
            logits_neg = feat_b @ neg.t() / tau
            maxv = torch.max(torch.cat([logits_pos, logits_neg], dim=1), dim=1, keepdim=True)[0]
            logits_pos = logits_pos - maxv; logits_neg = logits_neg - maxv
            loss_b = -torch.log(
                logits_pos.exp().sum() / (logits_pos.exp().sum() + logits_neg.exp().sum() + 1e-8)
            )
            total_loss += loss_b; valid += 1
        return total_loss / max(valid, 1)

    @staticmethod
    def _replace_token(tokens4, which, mode='random'):
        idx = 2 if which == 'g_spec' else 3
        rep = random_like_with_stats(tokens4[:, idx:idx + 1, :]) if mode == 'random' else torch.zeros_like(tokens4[:, idx:idx + 1, :])
        out = tokens4.clone(); out[:, idx:idx + 1, :] = rep; return out

    @torch.no_grad()
    def _append_to_bank(self, feats_4C, labels):
        if labels is None: return
        if torch.is_tensor(labels):
            labels = labels.view(-1).tolist()
        elif isinstance(labels, int):
            labels = [labels]
        B = feats_4C.shape[0]
        for b in range(B):
            lb = int(labels[b] if b < len(labels) else labels[-1])
            key = f"expert_bank_{lb}"
            cur = getattr(self, key)
            if cur.shape[0] < self.bank_length:
                new_bank = torch.cat([cur, feats_4C[b:b + 1]], dim=0)
            else:
                keep = max(0, cur.shape[0] - 1)
                new_bank = torch.cat([cur[-keep:], feats_4C[b:b + 1]], dim=0) if keep > 0 else feats_4C[b:b + 1]
            self._buffers[key] = new_bank.detach()

    # ================== 前向 ==================
    def forward(self, x_path, x_omic, phase, label=None, c=None, **kwargs):
        """
        x_path: [B, N, C_sp] - SPFusion �?token
        x_omic: [B, G]
        """
        out = {}
        assert torch.is_tensor(x_path) and x_path.dim() == 3, \
            f"x_path 需为 [B,N,C_sp]，收到 {type(x_path)} {getattr(x_path, 'shape', None)}"

        B_orig, N_orig, Csp = x_path.shape
        x_path, N_sampled, was_sampled = self._adaptive_token_sampling(x_path)

        if self.training and self.token_keep_ratio < 1.0:
            B_, N_, C_ = x_path.shape
            keep = max(int(N_ * self.token_keep_ratio), self.min_tokens)
            idx  = torch.randperm(N_, device=x_path.device)[:keep]
            x_path = x_path[:, idx, :]

        geno_feat = torch.stack([
            self.genomics_fc[idx](x_omic[..., length:self.genomics_idx[idx + 1]])
            for idx, length in enumerate(self.genomics_idx[:-1])
        ], dim=1)
        geno_indiv = self.geno_conv(self.geno_fc(geno_feat))

        z_tokens = self.path_token_proj(x_path)
        centers_full, viz_full, path_lb_loss = self.path_moe(
            z_tokens, geno_vec=geno_indiv, keep_topk_snapshot=(phase != 'train')
        )
        path_indiv = self.path_conv(self.path_fc(centers_full))

        tokens4, (aux_loss, aux_extras) = self.know_decompose(
            gfeat=geno_indiv, pfeat=path_indiv, return_losses=True
        )

        expert_feats_full, attn_full, attn_align_loss = self._expert_pool(tokens4, return_align_loss=True)
        expert_logits = torch.stack([
            self.expert_heads[i](expert_feats_full[:, i, :]) for i in range(self.num_interactions)
        ], dim=1)
        expert_probs = torch.sigmoid(expert_logits).clamp(1e-6, 1 - 1e-6)
        g = self.reweight_norm(tokens4.mean(dim=1))
        w = F.softmax(self.reweight_mlp(g), dim=-1)
        hazards_prob = (expert_probs * w.unsqueeze(-1)).sum(dim=1)
        S = torch.cumprod(1 - hazards_prob, dim=1)

        i2_losses = {}
        if phase == 'train':
            tokens_gabl = self._replace_token(tokens4, 'g_spec', mode='random')
            tokens_pabl = self._replace_token(tokens4, 'p_spec', mode='random')
            feats_gabl, _, _ = self._expert_pool(tokens_gabl, return_align_loss=False)
            feats_pabl, _, _ = self._expert_pool(tokens_pabl, return_align_loss=False)
            i2_losses = self._interaction_losses(expert_feats_full, feats_gabl, feats_pabl)

        con_loss = torch.tensor(0.0, device=hazards_prob.device)
        if phase == 'train' and (label is not None) and self.i2moe_w_con > 0:
            con_loss = self._supcon_with_bank(expert_feats_full.detach(), label)

        if phase == 'train' and (label is not None):
            self._append_to_bank(expert_feats_full.detach(), label)

        out['decompose'] = [tokens4, [geno_indiv, path_indiv]]
        banks_view = [getattr(self, f"expert_bank_{i}") for i in range(self.n_classes)]
        out['cohort'] = [banks_view, c]
        out['hazards'] = [hazards_prob]
        out['S'] = [S]
        out['losses'] = out.get('losses', {})
        out['losses']['fusion_aux'] = aux_loss
        if self.path_lb_weight > 0:
            out['losses']['path_load_balance'] = self.path_lb_weight * path_lb_loss
        if i2_losses:
            if 'i2moe_uni_g' in i2_losses: out['losses']['i2moe_uni_g'] = self.i2moe_w_uni * i2_losses['i2moe_uni_g']
            if 'i2moe_uni_p' in i2_losses: out['losses']['i2moe_uni_p'] = self.i2moe_w_uni * i2_losses['i2moe_uni_p']
            if 'i2moe_synergy' in i2_losses: out['losses']['i2moe_synergy'] = self.i2moe_w_syn * i2_losses['i2moe_synergy']
            if 'i2moe_redundancy' in i2_losses: out['losses']['i2moe_redundancy'] = self.i2moe_w_red * i2_losses['i2moe_redundancy']
            if self.i2moe_w_attn > 0: out['losses']['i2moe_attn_align'] = self.i2moe_w_attn * attn_align_loss
        if self.i2moe_w_con > 0:
            out['losses']['i2moe_supcon'] = self.i2moe_w_con * con_loss

        out['mi_metrics'] = aux_extras
        out['viz'] = {
            'token_stats': {
                'original_N': N_orig,
                'sampled_N': x_path.shape[1],
                'was_sampled': bool(was_sampled or (self.training and self.token_keep_ratio < 1.0)),
                'sampling_ratio': float(x_path.shape[1] / max(N_orig, 1)),
            },
            'path_moe': {
                'gate_mass': viz_full['gate_mass'].detach().cpu(),
                'counts': viz_full['counts'].detach().cpu(),
                'expert_usage': viz_full['expert_usage'].detach().cpu(),
                'avg_tokens_per_expert': viz_full['avg_tokens_per_expert'].detach().cpu(),
                'routing_confidence': viz_full['routing_confidence'],
                'topk_idx': viz_full.get('topk_idx', None),
            },
            'prediction_moe': {
                'attn_on_tokens': attn_full.detach().cpu(),
                'expert_weights': w.detach().cpu(),
                'expert_gate': torch.sigmoid(self.expert_residual_gate).detach().cpu(),
                'expert_prior': self.expert_prior_strength.detach().cpu(),
                'anchor_map': torch.tensor(self._anchor_map()),
            },
            'shapes': {
                'x_path': [B_orig, N_orig, Csp],
                'z_tokens': list(z_tokens.shape),
                'centers': list(centers_full.shape),
                'tokens4': list(tokens4.shape),
            },
        }
        return out


