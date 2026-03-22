import logging
import os

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from model.bert_model import BertPoolingForTripletPrediction
from model.data_collator import PoolingCollator
from model.data_processor import DictDataset, KGProcessor
from model.roberta_model import RobertaPoolingForTripletPrediction
from model.trainer import KGCTrainer
from model.utils import DataArguments, ModelArguments

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    raise KeyError(task_name)

def build_eval_dataset(features):
    """专家重构：将重复的 Dataset 构造代码抽象为通用函数，符合 DRY 原则"""
    dataset_dict = {
        "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
        "pos_indicator": torch.tensor([f.pos_indicator for f in features], dtype=torch.long),
    }
    # 动态检查是否含有结构特征
    if hasattr(features[0], "head_neighbors"):
        dataset_dict["head_neighbors"] = torch.stack([f.head_neighbors for f in features]).long()
        dataset_dict["tail_neighbors"] = torch.stack([f.tail_neighbors for f in features]).long()
        # 注意：mask由重构后的 data_collator 自动处理，这里只需传 neighbors
    return DictDataset(**dataset_dict)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.model_cache_dir)
    is_world_process_zero = training_args.local_rank == -1 or torch.distributed.get_rank() == 0

    processor = KGProcessor(data_args, tokenizer, is_world_process_zero)
    train_data, dev_data, test_data = processor.get_dataset(training_args)

    config = AutoConfig.from_pretrained(model_args.config_name or model_args.model_name_or_path, cache_dir=model_args.model_cache_dir)
    if not hasattr(config, "real_vocab_size"):
        config.real_vocab_size = config.vocab_size

    if model_args.pos_weight is not None:
        model_args.pos_weight = torch.tensor([model_args.pos_weight]).to(training_args.device)

    if model_args.pooling_model:
        logger.info("Using pooling model for Triplet Classification!")
        if tokenizer.__class__.__name__.startswith("Roberta"):
            tokenizer_cls = RobertaPoolingForTripletPrediction
        elif tokenizer.__class__.__name__.startswith("Bert"):
            tokenizer_cls = BertPoolingForTripletPrediction
        else:
            raise NotImplementedError()

        model = tokenizer_cls.from_pretrained(
            model_args.model_name_or_path,
            margin=data_args.margin,
            from_tf=bool(".ckpt" in model_args.model_name_or_path
