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
    dataset_dict = {
        "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
        "pos_indicator": torch.tensor(
            [f.pos_indicator for f in features], dtype=torch.long
        ),
    }
    if hasattr(features[0], "head_neighbors"):
        dataset_dict["head_neighbors"] = torch.stack(
            [f.head_neighbors for f in features]
        ).long()
        dataset_dict["tail_neighbors"] = torch.stack(
            [f.tail_neighbors for f in features]
        ).long()
    return DictDataset(**dataset_dict)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.model_cache_dir
    )
    is_world_process_zero = (
        training_args.local_rank == -1 or torch.distributed.get_rank() == 0
    )

    processor = KGProcessor(data_args, tokenizer, is_world_process_zero)
    train_data, dev_data, test_data = processor.get_dataset(training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
    )
    if not hasattr(config, "real_vocab_size"):
        config.real_vocab_size = config.vocab_size

    if model_args.pos_weight is not None:
        model_args.pos_weight = torch.tensor([model_args.pos_weight]).to(
            training_args.device
        )

    if model_args.pooling_model:
        logger.info("Using pooling model for Triplet Classification!")
        if tokenizer.__class__.__name__.startswith("Bert"):
            tokenizer_cls = BertPoolingForTripletPrediction
        else:
            raise NotImplementedError()

        model = tokenizer_cls.from_pretrained(
            model_args.model_name_or_path,
            margin=data_args.margin,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.model_cache_dir,
            pos_weight=model_args.pos_weight,
            text_loss_weight=model_args.text_loss_weight,
            use_structure=True,
            num_entities=processor.ent_size,
        )
        data_collator = PoolingCollator(tokenizer)
    else:
        raise NotImplementedError()

    trainer = KGCTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=dev_data,
        prediction_loss_only=True,
    )

    if data_args.group_shuffle:
        trainer.use_group_shuffle(data_args.num_neg)

    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

        if training_args.do_predict:
            logger.info("***** Running Triplet Classification *****")
            label_map = {"-1": 0, "1": 1}
            trainer.model.set_predict_mode()
            trainer.prediction_loss_only = False
            trainer.data_collator.set_predict_mode()

            dev_triples, dev_labels_raw = processor.get_dev_triples(return_label=True)
            dev_labels = np.array([label_map[l] for l in dev_labels_raw], dtype=int)
            _, dev_features = processor._create_examples_and_features(dev_triples)

            eval_data = build_eval_dataset(dev_features)
            dev_preds = trainer.predict(eval_data).predictions
            logger.info("Searching optimal threshold on Dev set...")
            thresholds = np.linspace(-5, 5, 1000)
            predictions_matrix = (dev_preds[:, None] - thresholds[None, :] > 0).astype(
                int
            )
            accuracies = np.mean(predictions_matrix == dev_labels[:, None], axis=0)

            best_idx = np.argmax(accuracies)
            max_acc = accuracies[best_idx]
            max_m = thresholds[best_idx]
            logger.info(
                f"Max Validation Accuracy: {max_acc:.4f} at threshold $\Delta$ = {max_m:.4f}"
            )

            test_triples, test_labels_raw = processor.get_test_triples(
                return_label=True
            )
            test_labels = np.array([label_map[l] for l in test_labels_raw], dtype=int)
            _, test_features = processor._create_examples_and_features(test_triples)

            test_data = build_eval_dataset(test_features)

            test_preds = trainer.predict(test_data).predictions
            final_test_preds = (test_preds - max_m > 0).astype(int)
            test_acc = np.mean(final_test_preds == test_labels)

            logger.info(f"Final Test Accuracy at threshold {max_m:.4f}: {test_acc:.4f}")


if __name__ == "__main__":
    main()
