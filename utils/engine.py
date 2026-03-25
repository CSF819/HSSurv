import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sksurv.metrics import concordance_index_censored
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

torch.set_num_threads(4)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.best_score = 0.0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler, subset):
        writer_dir = os.path.join(self.results_dir, f"{subset}_fold_{self.fold}")
        os.makedirs(writer_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(writer_dir, flush_secs=15)
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            model = model.cuda()

        if getattr(self.args, 'resume', None) is not None:
            if os.path.isfile(self.args.resume):
                print(f"=> loading checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint (score: {checkpoint['best_score']})")
            else:
                print(f"=> no checkpoint found at '{self.args.resume}'")

        if getattr(self.args, 'evaluate', False):
            self.run_epoch(val_loader, model, criterion, phase='eval')
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train
            self.run_epoch(train_loader, model, criterion, phase='train', optimizer=optimizer)
            # eval
            c_index = self.run_epoch(val_loader, model, criterion, phase='eval')

            # 保存最优
            if c_index >= self.best_score:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score,
                    'subset': subset,
                })

                # ★ 仅在刷新最优时：把刚刚 eval 缓存的 KM 数据写成 km_best.csv
                try:
                    if getattr(self, "_km_buffer", None) is not None:
                        km_dir = os.path.join(self.results_dir, f"{subset}_fold_{self.fold}", "km_exports")
                        os.makedirs(km_dir, exist_ok=True)
                        km_csv = os.path.join(km_dir, f"km_{subset}_fold{self.fold}_best.csv")
                        df = pd.DataFrame(self._km_buffer)
                        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        df.dropna(subset=["time", "event", "risk"], inplace=True)
                        df.to_csv(km_csv, index=False, encoding="utf-8")
                        print(f"[KM] 已保存最优 KM 到：{km_csv}")
                    else:
                        print("[KM] 跳过：_km_buffer 为空（本轮 eval 未缓存到 KM 数据）")
                except Exception as e:
                    print(f"[KM] 保存最优 KM 失败：{e}")

            print(f" *** best c-index={self.best_score:.4f} at epoch {self.best_epoch}")
            print('>')
        return self.best_score, self.best_epoch

   
    def run_epoch(self, data_loader, model, criterion, phase='train', optimizer=None):
        eval(f"model.{phase}()")

        sum_loss = 0.0
        risk_list, cens_list, time_list = [], [], []
        idx_list = []  # 用于还原 case_id

        # 统计
        num_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
        num_samples = len(getattr(data_loader, "dataset", [])) if hasattr(data_loader, "dataset") else None
        processed_samples = 0

        # 进度条
        pos = 0 if phase == 'train' else 1
        pbar = tqdm(
            data_loader,
            total=num_batches,
            desc=f"{phase} Epoch {self.epoch + 1}",
            dynamic_ncols=True,
            leave=True,
            position=pos
        )

        for batch_idx, (data_WSI, data_omic, label, event_time, c, idx) in enumerate(pbar):
            # 样本数累计
            bs = label.size(0) if hasattr(label, 'size') else (len(label) if hasattr(label, '__len__') else 0)
            processed_samples += bs

            use_cuda = torch.cuda.is_available()
            data_WSI = data_WSI.float().cuda() if use_cuda else data_WSI.float()
            data_omic = data_omic.float().cuda() if use_cuda else data_omic.float()
            label = label.float().cuda() if use_cuda else label.float()
            event_time = event_time.float().cuda() if use_cuda else event_time.float()
            c = c.float().cuda() if use_cuda else c.float()

            if phase == 'train':
                out = model(x_path=data_WSI, x_omic=data_omic, phase=phase, label=label, c=c)
                loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})

                if hasattr(self, 'writer') and self.writer and isinstance(out, dict):
                    viz = out.get('viz', {})
                    pm = viz.get('path_moe', None)
                    if pm is not None:
                        try:
                            total_batches = num_batches if num_batches is not None else (
                                len(data_loader) if hasattr(data_loader, "__len__") else 0)
                            global_step = int(self.epoch) * int(total_batches) + int(batch_idx)
                        except Exception:
                            global_step = self.epoch

                        rc = pm.get('routing_confidence', None)
                        if rc is not None:
                            try:
                                rc_val = float(rc.detach().cpu().item()) if isinstance(rc, torch.Tensor) else float(rc)
                                self.writer.add_scalar(f'{phase}/path_moe/routing_confidence', rc_val, global_step)
                            except Exception:
                                pass

                        eu = pm.get('expert_usage', None)
                        if eu is not None:
                            try:
                                if isinstance(eu, torch.Tensor):
                                    eu_arr = eu.detach().cpu().numpy()
                                elif isinstance(eu, np.ndarray):
                                    eu_arr = eu
                                else:
                                    eu_arr = np.asarray(eu)
                                eu_var = float(np.var(eu_arr))
                                self.writer.add_scalar(f'{phase}/path_moe/expert_usage_var', eu_var, global_step)
                            except Exception:
                                pass
                # === end ===

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_val = optimizer.param_groups[-1]['lr']
            else:
                with torch.no_grad():
                    out = model(x_path=data_WSI, x_omic=data_omic, phase=phase)
                    loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})
                lr_val = 0.0

            sum_loss += loss.item()
            avg_loss = sum_loss / (batch_idx + 1)

            risks = -torch.sum(out['S'][-1], dim=1).detach().cpu().numpy()
            risk_list.append(risks)
            cens_list.append(c.detach().cpu().numpy().reshape(-1))
            time_list.append(event_time.detach().cpu().numpy().reshape(-1))

            if torch.is_tensor(idx):
                idx_list.append(idx.detach().cpu().numpy().reshape(-1))
            else:
                idx_list.append(np.asarray(idx).reshape(-1))

            batch_prog = f"[{batch_idx + 1}/{num_batches}]" if num_batches is not None else f"[{batch_idx + 1}]"
            sample_prog = (f"({processed_samples}/{num_samples} samples)" if num_samples is not None
                           else f"({processed_samples} samples)")
            pbar.set_postfix_str(
                f"{batch_prog} {sample_prog} | LR: {lr_val:.1e} | loss: {loss.item():.4f} | avg: {avg_loss:.4f}"
            )

        # ====== epoch 摘要 ======
        mean_loss = sum_loss / max(1, (num_batches if num_batches is not None else (batch_idx + 1)))
        all_risk_scores = np.concatenate(risk_list, axis=0)
        all_censorships = np.concatenate(cens_list, axis=0)
        all_event_times = np.concatenate(time_list, axis=0)
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
        )[0]

        tqdm.write(f"[{phase}] Epoch {self.epoch + 1} | loss: {mean_loss:.4f}, c_index: {c_index:.4f}")

        if hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar(f'{phase}/loss', mean_loss, self.epoch)
            self.writer.add_scalar(f'{phase}/c_index', c_index, self.epoch)

        if phase == 'eval':
            try:
                all_idx = np.concatenate(idx_list, axis=0)
                ds = getattr(data_loader, "dataset", None)
                if hasattr(ds, "slide_data") and ("case_id" in getattr(ds.slide_data, "columns", [])):
                    case_ids = ds.slide_data["case_id"].iloc[all_idx].astype(str).tolist()
                elif hasattr(ds, "case_ids"):
                    case_ids = np.asarray(ds.case_ids)[all_idx].astype(str).tolist()
                else:
                    case_ids = [f"idx_{int(i)}" for i in all_idx]

                self._km_buffer = {
                    "case_id": case_ids,
                    "time": all_event_times,
                    "event": (1 - all_censorships).astype(int),
                    "risk": all_risk_scores,
                    "dataset": getattr(self.args, "dataset", str(subset)) if 'subset' in locals() else getattr(
                        self.args, "dataset", ""),
                    "fold": int(self.fold),
                    "epoch": int(self.epoch + 1),
                }
            except Exception as e:
                self._km_buffer = None
                print(f"[KM] 缓存失败：{e}")

        return c_index

    def save_checkpoint(self, state):
        # 确保 ckpt 子目录存在
        ckpt_dir = os.path.join(self.results_dir, f"{state['subset']}_fold_{self.fold}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 删除上一份最优（如果存在）
        if self.filename_best is not None:
            try:
                os.remove(self.filename_best)
            except Exception:
                pass

        # 模型文件名：带上 fold 信息
        self.filename_best = os.path.join(
            ckpt_dir,
            'model_best_{score:.4f}_epoch{epoch}_fold{fold}.pth.tar'.format(
                score=state['best_score'],
                epoch=state['epoch'],
                fold=self.fold
            )
        )
        print(f'save best model {self.filename_best}')
        torch.save(state, self.filename_best)

