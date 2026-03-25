import os
import sys
import csv
import time
import random
import torch
import numpy as np
import importlib

from shutil import copyfile
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from utils.options import parse_args
from utils.loss_factory_new import Loss_factory
from utils.engine import Engine
from utils.dataset_survival import Generic_MIL_Survival_Dataset


# ---------------- 工具：种子固定 ----------------
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ====== 配置快照工具：不改训练，仅做I/O ======
import json
from copy import deepcopy

def _namespace_to_pure_dict(ns):
    """argparse.Namespace -> 纯字典（可JSON序列化）"""
    if isinstance(ns, dict):
        return {k: _namespace_to_pure_dict(v) for k, v in ns.items()}
    if hasattr(ns, "__dict__"):
        return _namespace_to_pure_dict(vars(ns))
    if isinstance(ns, (list, tuple)):
        return [_namespace_to_pure_dict(v) for v in ns]
    # numpy/tensor 等转基本类型
    try:
        import numpy as _np
        import torch as _torch
        if isinstance(ns, _np.generic):
            return _np.asscalar(ns)
        if isinstance(ns, _np.ndarray):
            return ns.tolist()
        if isinstance(ns, _torch.Tensor):
            return ns.detach().cpu().tolist()
    except Exception:
        pass
    return ns

def _collect_moe_config(model):
    moe = {}
    try:
        # 病理侧 Token MoE
        pm = getattr(model, "path_moe", None)
        if pm is not None:
            moe["path_moe"] = {
                "in_dim": getattr(pm, "C", None),
                "num_experts": getattr(pm, "K", None),
                "top_k": getattr(pm, "top_k", None),
                "use_geno_in_gate": getattr(pm, "use_geno_in_gate", None),
                "gate_geno_ratio": getattr(pm, "gate_geno_ratio", None),
                "load_balance_weight": getattr(pm, "load_balance_weight", None),
                "lb_hard_usage": getattr(pm, "lb_hard_usage", None),
                "gate_temperature": getattr(pm, "gate_temperature", None),
                "gate_noise_std": getattr(pm, "gate_noise_std", None),
            }
        # 模型级别的稳健化旋钮
        moe["model_level"] = {
            "spfusion_dim": getattr(model, "C_sp", None),
            "feat_dim": getattr(model, "feat_dim", None),
            "token_keep_ratio": getattr(model, "token_keep_ratio", None),
            "attn_temp": getattr(model, "attn_temp", None),
            "expert_dropout": getattr(model, "expert_dropout", None),
            "bank_length": getattr(model, "bank_length", None),
            "i2moe_w_uni": getattr(model, "i2moe_w_uni", None),
            "i2moe_w_syn": getattr(model, "i2moe_w_syn", None),
            "i2moe_w_red": getattr(model, "i2moe_w_red", None),
            "i2moe_w_attn": getattr(model, "i2moe_w_attn", None),
            "i2moe_w_con": getattr(model, "i2moe_w_con", None),
            "i2moe_tau": getattr(model, "i2moe_tau", None),
            "path_load_balance_weight": getattr(model, "path_lb_weight", None),
        }
    except Exception as e:
        moe["error"] = f"collect_moe_failed: {e}"
    return moe

def _save_run_config(results_dir, dataset, fold, args, model, net_module_path=None):
    os.makedirs(results_dir, exist_ok=True)
    snap_dir = os.path.join(results_dir, f"{dataset}_fold_{fold}")
    os.makedirs(snap_dir, exist_ok=True)

    # 1) 原样文本
    txt_path = os.path.join(snap_dir, "run_args.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(args) + "\n")

    # 2) JSON 版
    json_path = os.path.join(snap_dir, "run_config.json")
    args_dict = _namespace_to_pure_dict(args)
    moe_dict = _collect_moe_config(model)
    payload = {
        "dataset": dataset,
        "fold": fold,
        "args": deepcopy(args_dict),
        "moe_snapshot": moe_dict,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 3) 复制当次用到的网络定义文件，确保代码留档
    if net_module_path and os.path.isfile(net_module_path):
        try:
            dst = os.path.join(snap_dir, os.path.basename(net_module_path))
            copyfile(net_module_path, dst)
        except Exception as e:
            print(f"[warn] 复制网络文件失败：{e}")

    print(f"[Config] 已保存配置到：{snap_dir}")

# ---------------- 工具：解析命令行显式覆盖的参数 ----------------
def _extract_cli_overrides(argv):
    """
    返回一个集合：用户在命令行里显式传入的 --xxx（转成 argparse 的属性名 xxx）
    例如: --lr 1e-4 --bank_length 16 -> {'lr','bank_length'}
    """
    flags = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            name = tok.lstrip("-")
            # argparse: 连字符会转下划线
            name = name.replace("-", "_")
            # 处理布尔开关：下一个可能是值也可能是下一个flag
            flags.add(name)
            # 跳过值（如果不是下一个flag/文件结尾）
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
                continue
        i += 1
    return flags


# ---------------- 工具：把 auto_config 写回 args（只填充未在命令行显式覆盖的项） ----------------
def _apply_auto_config_to_args(args, auto_cfg, protected_keys):
    """
    args: argparse.Namespace
    auto_cfg: {'base': {...}, 'customized': {k: {'type':T,'default':v}}}
    protected_keys: set, 命令行显式传入的键（不覆盖）
    """
    # base
    for k, v in auto_cfg.get('base', {}).items():
        if k not in protected_keys:
            setattr(args, k, v)

    # customized
    for k, spec in auto_cfg.get('customized', {}).items():
        v = spec.get('default', None)
        if k not in protected_keys:
            setattr(args, k, v)

    return args


def main(args):
    print(args)

    # 解析 fold 列表，并构造统一的 fold 后缀（单 fold: fold0；多 fold: fold0+1+2）
    folds = list(map(int, args.fold.split(',')))
    fold_suffix = "fold" + "+".join(str(f) for f in folds)

    datalist = args.sets.split(',')
    all_best_score = {}

    # 记录命令行显式覆盖项
    cli_overrides = _extract_cli_overrides(sys.argv[2:])  # 第一个是脚本名，第二个是model名

    for dataname in datalist:
        # —— 每个数据集一个完整 run —— #
        args.dataset = dataname.lower()

        # === 每个数据集独立目录：数据集名_模型名_训练时间_折信息 ===
        run_time = time.strftime("[%H-%M-%S]")
        results_dir = os.path.join("./results", f"{dataname}_{args.model}_{run_time}_{fold_suffix}")
        os.makedirs(results_dir, exist_ok=True)

        # 针对该数据集的结果 CSV（文件名也带 fold 后缀）
        csv_path = os.path.join(results_dir, f"results_level_{args.level}_{fold_suffix}.csv")
        header = ["name"] + [f"fold {f}" for f in folds] + ["mean", "std"]
        print("############", csv_path)
        with open(csv_path, "a+", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)

        # === 加载模型模块，并对当前数据集应用 auto-config ===
        # 注意：不改变训练策略，只填充默认超参到 args（命令行已给出的不动）
        try:
            net_mod = importlib.import_module(f'models.{args.model}.network')
        except Exception as e:
            raise NotImplementedError(f"加载模型模块失败 models.{args.model}.network: {e}")

        if hasattr(net_mod, 'get_auto_config'):
            auto_cfg = net_mod.get_auto_config(dataset_name=args.dataset)
            args = _apply_auto_config_to_args(args, auto_cfg, protected_keys=cli_overrides)
            # 仅提示关键自适应项（便于日志对齐复现实验）
            print(f"[AutoConfig] dataset={args.dataset} "
                  f"experts={getattr(args,'path_num_experts', None)}, "
                  f"topk={getattr(args,'path_topk', None)}, "
                  f"dropout={getattr(args,'expert_dropout', None)}, "
                  f"attn_temp={getattr(args,'attn_temp', None)}, "
                  f"lb_w={getattr(args,'path_load_balance_weight', None)}, "
                  f"bank_len={getattr(args,'bank_length', None)}, "
                  f"lr={getattr(args,'lr', None)}, "
                  f"weight_decay={getattr(args,'weight_decay', None)}")
        else:
            print("[Warn] 模型模块未暴露 get_auto_config，跳过自动配置。")

        # 记录每个 fold 的最佳 epoch 和得分
        best_epoch_row = ["best epoch"]
        best_score_row = [dataname]
        fold_scores = []

        # ====== K-fold CV ======
        for fold in folds:
            set_seed(args.seed)

            dataset = Generic_MIL_Survival_Dataset(
                csv_path=rf"E:\WSI\csv\tcga_{dataname}_all_clean_filtered.csv",
                modal=args.modal,
                apply_sig=True,
                data_dir=args.data_root_dir,
                shuffle=False,
                seed=args.seed,
                patient_strat=False,
                n_bins=4,
                label_col="survival_months",
            )
            split_dir = f"./splits/{args.which_splits}/tcga_{dataname}_new"
            train_dataset, val_dataset = dataset.return_splits(
                from_id=False,
                csv_path=f"{split_dir}/splits_{fold}.csv",
                set_name=args.dataset,
                extractor=args.extractor
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            print(f"Dataset {dataname} split_{fold}: train {len(train_dataset)}, val {len(val_dataset)}")

            # ====== 构建模型与优化器/调度器 ======
            try:
                # 与上面相同模块，构建类名由 args.model 指定（如 HSSurv）
                # ====== 构建模型与优化器/调度器 ======
                model_class = getattr(net_mod, args.model)
                model = model_class(args)

                # ★ 保存当次运行的配置与 MoE 快照（只做记录，不改训练）
                try:
                    # 还原网络文件的真实路径，便于复制留档
                    net_file_path = None
                    if hasattr(net_mod, "__file__"):
                        net_file_path = net_mod.__file__
                    _save_run_config(results_dir, dataname, fold, args, model, net_module_path=net_file_path)
                except Exception as e:
                    print(f"[Config] 保存失败但不影响训练：{e}")

            except Exception as e:
                raise NotImplementedError(f"实例化模型 '{args.model}' 失败: {e}")

            base_lr = float(getattr(args, "lr", 2e-4))
            weight_decay = float(getattr(args, "weight_decay", 1e-4))
            optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

            total_epochs = int(getattr(args, "num_epoch", getattr(args, "max_epoch", 100)))
            warmup_epochs = int(getattr(args, "warmup_epochs", 3))
            T_max = max(int(total_epochs) - int(warmup_epochs), 1)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6),
                ],
                milestones=[warmup_epochs],
            )

            engine = Engine(args, results_dir, fold)  # 传入该数据集的独立目录
            criterion = Loss_factory(args)

            # ====== 开始训练 ======
            fold_start_time = time.time()
            score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler, dataname)
            fold_end_time = time.time()
            elapsed_time = fold_end_time - fold_start_time
            print(f"Fold {fold} 总训练耗时：{elapsed_time:.2f} 秒（约 {elapsed_time / 60:.2f} 分钟）")

            best_epoch_row.append(epoch)
            best_score_row.append(score)
            fold_scores.append(score)

        # ====== 汇总统计并写回 CSV ======
        mean_score = float(np.mean(fold_scores)) if len(fold_scores) > 0 else 0.0
        std_score = float(np.std(fold_scores)) if len(fold_scores) > 0 else 0.0
        best_score_row += [mean_score, std_score]

        all_best_score[dataname] = fold_scores

        print("############", csv_path)
        with open(csv_path, "a+", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(best_epoch_row)
            writer.writerow(best_score_row)

    print(all_best_score)
    return all_best_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()

    args = parse_args(sys.argv[1])
    main(args)
    print("Finished!")
