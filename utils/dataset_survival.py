from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 分层注意力融合模块（自动适配维度）
# ============================================================
class HierarchicalAttentionFusion(nn.Module):
    """
    利用 slide 全局特征进行分层注意力筛选：
    1. 粗筛选：slide 指导选出 top-k% 重要 patch
    2. 细筛选：交叉注意力精细化筛选
    3. 多尺度融合：全局 + 粗尺度 + 细尺度
    """

    def __init__(self,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 topk_ratio=0.3,
                 use_learnable_pos=True):
        super().__init__()

        self.topk_ratio = topk_ratio
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_learnable_pos = use_learnable_pos

        # 延迟初始化标志
        self._initialized = False
        self.slide_dim = None
        self.patch_dim = None
        self.output_dim = None

    def _lazy_init(self, slide_dim, patch_dim):
        """根据输入维度延迟初始化网络层"""
        if self._initialized:
            return

        self.slide_dim = slide_dim
        self.patch_dim = patch_dim
        self.output_dim = patch_dim  # 输出维度与 patch 一致
        d = self.output_dim

        dev = next(self.parameters()).device if len(list(self.parameters())) > 0 else device

        # === 投影层 ===
        self.proj_slide = nn.Sequential(
            nn.Linear(slide_dim, d),
            nn.LayerNorm(d)
        ).to(dev)

        self.proj_patch = nn.Sequential(
            nn.Linear(patch_dim, d),
            nn.LayerNorm(d)
        ).to(dev)

        # === 类型嵌入 ===
        self.type_slide = nn.Parameter(torch.zeros(1, 1, d, device=dev))
        self.type_patch = nn.Parameter(torch.zeros(1, 1, d, device=dev))

        # === 第一层：粗粒度注意力（全局筛选）===
        self.coarse_attn = nn.Sequential(
            nn.Linear(d * 2, d // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(d // 2, 1)
        ).to(dev)

        # === 第二层：细粒度交叉注意力 ===
        self.fine_cross_attn = nn.MultiheadAttention(
            d, self.num_heads, dropout=self.dropout, batch_first=True
        ).to(dev)

        self.fine_norm = nn.LayerNorm(d).to(dev)

        # === 第三层：Transformer Encoder 细化 ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.num_heads,
            dim_feedforward=d * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers).to(dev)

        # === 第四层：门控MIL池化 ===
        self.mil_pool = GatedMILPooling(d).to(dev)

        # === 多尺度融合 ===
        self.scale_fusion = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d)
        ).to(dev)

        self._initialized = True
        self.d = d  # 保存给外部访问

    def forward(self, slide_feat, patch_feats, patch_coords=None, patch_mask=None, return_tokens=True):
        """
        slide_feat: [slide_dim] 或 [1, slide_dim]
        patch_feats: [N, patch_dim] 或 [1, N, patch_dim]
        """
        # 维度处理
        if slide_feat.dim() == 1:
            slide_feat = slide_feat.unsqueeze(0)
        if patch_feats.dim() == 2:
            patch_feats = patch_feats.unsqueeze(0)

        B, N, patch_dim = patch_feats.shape
        slide_dim = slide_feat.shape[-1]

        # 延迟初始化
        if not self._initialized:
            self._lazy_init(slide_dim, patch_dim)

        # 投影
        s = self.proj_slide(slide_feat).unsqueeze(1) + self.type_slide  # [B, 1, d]
        p = self.proj_patch(patch_feats) + self.type_patch  # [B, N, d]

        # ===== 阶段1：粗筛选（slide 指导的全局注意力）=====
        s_expand = s.expand(-1, N, -1)  # [B, N, d]
        coarse_input = torch.cat([p, s_expand], dim=-1)  # [B, N, 2d]
        coarse_scores = self.coarse_attn(coarse_input).squeeze(-1)  # [B, N]

        # 选出 top-k
        k = max(1, int(N * self.topk_ratio))
        topk_vals, topk_idx = torch.topk(coarse_scores, k=k, dim=1)

        # 提取 top-k patch
        b_idx = torch.arange(B, device=patch_feats.device).unsqueeze(-1)
        p_topk = p[b_idx, topk_idx]  # [B, k, d]

        # ===== 阶段2：细筛选（交叉注意力）=====
        s_query = s  # [B, 1, d]
        fine_out, fine_weights = self.fine_cross_attn(
            s_query, p_topk, p_topk, need_weights=True
        )  # [B, 1, d]

        s_refined = s + fine_out  # 残差连接
        s_refined = self.fine_norm(s_refined)

        # ===== 阶段3：Transformer 全局交互 =====
        tokens = torch.cat([s_refined, p], dim=1)  # [B, 1+N, d]
        tokens = self.encoder(tokens)

        s_final = tokens[:, 0, :]  # [B, d]
        p_final = tokens[:, 1:, :]  # [B, N, d]

        # ===== 阶段4：门控MIL池化 =====
        p_pooled, _ = self.mil_pool(p_final, mask=patch_mask)  # [B, d]

        # ===== 阶段5：多尺度融合 =====
        # 全局特征
        global_feat = s_final
        # 粗尺度特征（top-k 平均）
        coarse_feat = p_topk.mean(dim=1)
        # 细尺度特征（门控池化）
        fine_feat = p_pooled

        # 拼接并融合
        multi_scale = torch.cat([global_feat, coarse_feat, fine_feat], dim=-1)
        fused = self.scale_fusion(multi_scale)  # [B, d]

        if return_tokens:
            return fused, p_final  # 返回更新后的 patch tokens
        return fused


class GatedMILPooling(nn.Module):
    """门控注意力池化"""

    def __init__(self, d):
        super().__init__()
        dh = max(1, d // 2)
        self.att_a = nn.Linear(d, dh)
        self.att_b = nn.Linear(dh, 1)
        self.gate = nn.Linear(d, dh)

    def forward(self, x, mask=None):
        """
        x: [B, N, D]
        mask: [B, N], True=有效
        """
        B, N, D = x.shape
        a = torch.tanh(self.att_a(x))
        g = torch.sigmoid(self.gate(x))
        h = a * g

        score = self.att_b(h).squeeze(-1)  # [B, N]

        if mask is not None:
            very_neg = torch.finfo(x.dtype).min / 2
            score = torch.where(mask, score, very_neg)
            valid_count = mask.sum(dim=1)
            zero_case = (valid_count == 0)
            if zero_case.any():
                pooled = torch.zeros(B, D, device=x.device, dtype=x.dtype)
                nz = ~zero_case
                if nz.any():
                    attn_nz = torch.softmax(score[nz], dim=1)
                    pooled[nz] = torch.einsum('bn,bnd->bd', attn_nz, x[nz])
                return pooled, None

        attn = torch.softmax(score, dim=1)
        pooled = torch.einsum('bn,bnd->bd', attn, x)
        return pooled, attn


# ============================================================
# Dataset 类（保持原有结构不变）
# ============================================================
class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv', modal='omic', apply_sig=False,
                 shuffle=False, seed=7, n_bins=4, ignore=[],
                 patient_strat=False, label_col=None, filter_dict={}, eps=1e-6, **kwargs):
        self.custom_test_ids = None
        self.seed = seed
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.fusion_model = None

        slide_data = pd.read_csv(csv_path, low_memory=False)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data.values)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code'].astype(str).values:
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.modal = modal
        self.cls_ids_prep()

        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('F:/PythonProjects/MCAT-master/datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def get_split_from_df(self, all_splits: dict, split_key: str = 'train', scaler=None, set_name='BLCA',
                          extractor='resnet50'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True).tolist()

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, modal=self.modal, signatures=self.signatures,
                                  data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict,
                                  num_classes=self.num_classes, set_name=set_name, extractor=extractor)
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None, set_name='BLCA', extractor='resnet50'):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train', set_name=set_name,
                                                 extractor=extractor)
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val', set_name=set_name,
                                               extractor=extractor)

            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, modal: str = 'omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.modal = modal
        self.use_h5 = False
        self.fusion_model = None

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if 'path' in self.modal or 'coattn' in self.modal or 'multi' in self.modal:
                    p_tokens_list = []
                    fused_global_last = None

                    for slide_id in slide_ids:
                        slide_path = os.path.normpath(os.path.join(data_dir, 'slide', f"{slide_id.rstrip('.svs')}.pt"))
                        patch_path = os.path.normpath(os.path.join(data_dir, 'patch', f"{slide_id.rstrip('.svs')}.pt"))
                        try:
                            slide_bag = torch.load(slide_path, map_location='cpu')
                            patch_bag = torch.load(patch_path, map_location='cpu')

                            # === 使用分层注意力融合（自动适配维度）===
                            if self.fusion_model is None:
                                self.fusion_model = HierarchicalAttentionFusion(
                                    num_heads=8,
                                    num_layers=2,
                                    dropout=0.1,
                                    topk_ratio=0.3,  # 保留 30% 最重要的 patch
                                    use_learnable_pos=True
                                ).to(device)

                            slide_bag = slide_bag.to(device)
                            patch_bag = patch_bag.to(device)

                            fused_vec, p_tokens = self.fusion_model(slide_bag, patch_bag, return_tokens=True)
                            fused_global_last = fused_vec.detach().cpu()
                            p_tokens_list.append(p_tokens.squeeze(0).detach().cpu())
                        except FileNotFoundError:
                            continue

                    if len(p_tokens_list) > 0:
                        path_features = torch.cat(p_tokens_list, dim=0)
                    else:
                        path_features = torch.zeros((1, 1))

                    # 数值消毒
                    if isinstance(path_features, torch.Tensor) and path_features.ndim == 2:
                        path_features = torch.clamp(path_features, -1e6, 1e6)
                        path_features = torch.nan_to_num(path_features, nan=0.0, posinf=1e6, neginf=-1e6)

                        # K 兜底
                        K_min = 6
                        N, D = path_features.shape
                        if 1 <= N < K_min:
                            need = K_min - N
                            idx_rep = torch.randint(low=0, high=N, size=(need,))
                            noise = 1e-6 * torch.randn(need, D)
                            path_features = torch.cat([path_features, path_features[idx_rep] + noise], dim=0)

                if 'omic' in self.modal:
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                elif 'coattn' in self.modal:
                    genomic_features = torch.cat(
                        [torch.tensor(self.genomic_features[omic_name].iloc[idx]) for omic_name in self.omic_names],
                        dim=0)
                elif 'multi' in self.modal:
                    genomic_features = torch.cat(
                        [torch.tensor(self.genomic_features[omic_name].iloc[idx]) for omic_name in self.omic_names],
                        dim=0)
                else:
                    genomic_features = torch.zeros((1, 1))

                return path_features, genomic_features, label, event_time, c, idx
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, modal, signatures=None, data_dir=None, label_col=None, patient_dict=None,
                 num_classes=2, set_name='BLCA', extractor='resnet50'):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.modal = modal
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.dataset = set_name
        self.extractor = extractor
        self.fusion_model = None

        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic + modal for modal in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple = None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed