import argparse
import importlib
import os

def parse_args(model_name):
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("model", type=str, default="HSSurv", help="Type of model (Default: HSSurv)")
    parser.add_argument("--data_root_dir", type=str, default="E:/WSI/Feather_Features/{}", help="Data directory to WSI features (extracted via CLAM)")
    parser.add_argument("--extractor", type=str, default="resnet50", help="Path to latest checkpoint (default: none)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible experiment (default: 1)")
    parser.add_argument("--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv)")
    parser.add_argument("--sets", type=str, default="luad,ucec,blca,brca,gbmlgg", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    parser.add_argument("--level", type=str, default="x20", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="Path to latest checkpoint (default: none)")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')


    # Model Parameters.
    parser.add_argument("--model_size", type=str, choices=["small", "large"], default="small", help="Size of some models (Transformer)")
    parser.add_argument("--modal", type=str, choices=["omic", "path", "coattn", "multi"], default="coattn", help="Specifies which modalities to use / collate function in dataloader.")
    parser.add_argument("--fusion", type=str, choices=["concat", "bilinear"], default="concat", help="Modality fuison strategy")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead", "SGD"], default="SGD")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nllsurv", help="slide-level classification loss function (default: ce)")

   # ====== 训练调度 & 稳定性 ======
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='线性 warm-up 轮数')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                        help='梯度裁剪阈值（0 关闭）')
    parser.add_argument('--detect_anomaly', type=int, default=0,
                        help='开启 autograd 异常检测（1 开启，训练更慢，仅排错用）')


    # Model-specific Parameters
    # model_specific_config = importlib.import_module('models.{}.network'.format(model_name)).custom_config
    model_specific_config = importlib.import_module(
        f"models.HSSurv.network"
    ).custom_config
    ### Base arguments with customized values
    parser.set_defaults(**model_specific_config['base'])

    ### Customized arguments
    for k, v in model_specific_config['customized'].items():
        v['dest'] = k
        parser.add_argument('--' + k, **v)

    args = parser.parse_args()
    # —— 规范化 loss 字符串，并按需自动追加 mi_ 权重 ——
    # 统一小写、去空格
    args.loss = ','.join([tok.strip().lower() for tok in args.loss.split(',') if tok.strip()])

    return args

    #return args--
