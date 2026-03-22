import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import EvalPrediction, PredictionOutput

logger = logging.getLogger(__name__)


class GroupRandomSampler(Sampler):
    def __init__(self, data_source, group):
        self.data_source = data_source
        self.group = group
        assert len(data_source) % group == 0

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source) // self.group
        perm = torch.randperm(n)
        perm = (perm[:, None] * self.group + torch.arange(self.group)).view(-1)
        return iter(perm.tolist())

    def __len__(self):
        return self.num_samples


class GroupDistributedSampler(DistributedSampler):
    def __init__(
        self, dataset, group_size, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        self.group_size = group_size
        assert len(dataset) % group_size == 0, (
            f"length of dataset should be a multiple of group size, but get length {len(dataset)} and group size {group_size}"
        )
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_groups = math.ceil(
            len(self.dataset) / self.group_size / self.num_replicas
        )
        self.num_samples = self.num_groups * self.group_size
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset) // self.group_size
            perm = torch.randperm(n, generator=g)
            perm = (
                perm[:, None] * self.group_size + torch.arange(self.group_size)
            ).view(-1)
            indices = perm.tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[: self.total_size - len(indices)]
        assert len(indices) == self.total_size

        # 专家优化：利用张量维度变换瞬间完成分布式数据切分，替代缓慢的 for 循环
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        indices_tensor = indices_tensor.view(
            self.num_groups, self.num_replicas, self.group_size
        )
        sub_indices = indices_tensor[:, self.rank, :].flatten().tolist()

        assert len(sub_indices) == self.num_samples
        return iter(sub_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class KGCTrainer(Trainer):
    def __init__(
        self, *args, structure_loss_weight=0.3, reconstruction_loss_weight=0.2, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.w_sem = 0.5
        self.w_str = structure_loss_weight
        self.w_rec = reconstruction_loss_weight

    def _compute_gated_loss(self, semantic_loss, structure_loss=None, recon_loss=None):
        """静态权重融合（建议后续把融合逻辑内聚到 Model 的 forward 中）"""
        total_loss = self.w_sem * semantic_loss
        if structure_loss is not None:
            total_loss += self.w_str * structure_loss
        if recon_loss is not None:
            total_loss += self.w_rec * recon_loss
        return total_loss

    def training_step(self, model, inputs):
        """重写训练步骤，兼容 FP16 与多损失"""
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)

        if isinstance(outputs, tuple) and len(outputs) >= 3:
            semantic_loss, structure_loss = outputs[0], outputs[1]
            recon_loss = outputs[2] if len(outputs) > 3 else None
            loss = self._compute_gated_loss(semantic_loss, structure_loss, recon_loss)
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else outputs

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 专家修复：兼容 transformers 3.0.2 的 FP16 逻辑
        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex to use fp16 training.")
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def use_group_shuffle(self, num_neg):
        self.group_shuffle = True
        self.num_neg = num_neg

    def stop_group_shuffle(self):
        self.group_shuffle = False

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            self._get_group_sampler()
            if getattr(self, "group_shuffle", False)
            else self._get_train_sampler()
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if hasattr(self, "group_shuffle") and self.group_shuffle:
            if self.args.local_rank != -1:
                sampler = GroupDistributedSampler(
                    eval_dataset, group_size=self.num_neg * 3 + 1, shuffle=False
                )
            else:
                sampler = SequentialSampler(eval_dataset)
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
        return data_loader

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
        return data_loader

    def _get_group_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                GroupRandomSampler(self.train_dataset, self.num_neg * 3 + 1)
                if self.args.local_rank == -1
                else GroupDistributedSampler(
                    self.train_dataset, group_size=self.num_neg * 3 + 1
                )
            )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.prediction_loss_only
        )
        model = torch.nn.DataParallel(self.model) if self.args.n_gpu > 1 else self.model

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", dataloader.batch_size)

        eval_losses: List[float] = []

        # 专家修复：使用 List 收集 Tensor，防止 OOM 显存溢出与二次方拼接降速
        preds_list = []
        labels_list = []

        model.eval()
        past = None

        for inputs in tqdm(
            dataloader,
            desc=description,
            disable=not getattr(self, "is_local_master", lambda: True)(),
        ):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                outputs = model(**inputs)

                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    logits = outputs[-1]
                    if has_labels:
                        eval_losses.append(outputs[0].mean().item())
                else:
                    if has_labels:
                        step_eval_loss, logits = outputs[:2]
                        eval_losses.append(step_eval_loss.mean().item())
                    else:
                        logits = outputs[0]

                if self.args.past_index >= 0:
                    past = outputs[
                        self.args.past_index if has_labels else self.args.past_index - 1
                    ]

            if not prediction_loss_only:
                # 关键修复：立即移至 CPU 并缓存，释放宝贵的 GPU 显存
                preds_list.append(logits.detach().cpu())
                if inputs.get("labels") is not None:
                    labels_list.append(inputs["labels"].detach().cpu())

        # 循环结束后统一执行 Cat，速度极快
        preds = torch.cat(preds_list, dim=0) if len(preds_list) > 0 else None
        label_ids = torch.cat(labels_list, dim=0) if len(labels_list) > 0 else None

        # 如果在分布式环境下，需额外收集
        if self.args.local_rank != -1:
            if preds is not None:
                preds = self.distributed_concat(
                    preds.to(self.args.device),
                    num_total_examples=self.num_examples(dataloader),
                ).cpu()
            if label_ids is not None:
                label_ids = self.distributed_concat(
                    label_ids.to(self.args.device),
                    num_total_examples=self.num_examples(dataloader),
                ).cpu()

        if preds is not None:
            preds = preds.numpy()
        if label_ids is not None:
            label_ids = label_ids.numpy()

        metrics = {}
        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )

        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
