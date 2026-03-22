import logging
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm
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
    else:
        raise KeyError(task_name)


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
        logger.info("Using pooling model!")
        if tokenizer.__class__.__name__.startswith("Roberta"):
            tokenizer_cls = RobertaPoolingForTripletPrediction
        elif tokenizer.__class__.__name__.startswith("Bert"):
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
            use_structure=True,  # 专家提醒：确保你的模型开启了结构信息
            num_entities=processor.ent_size,  # 传入实体数量用于构建 Embedding
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
        logger.info("Using group shuffle")
        trainer.use_group_shuffle(data_args.num_neg)

    # ==========================
    # Train & Eval Pipeline
    # ==========================
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

    if training_args.do_eval:
        eval_output = trainer.evaluate()
        if trainer.is_world_master():
            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results_lm.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

    # ==========================
    # Link Prediction Pipeline
    # ==========================
    if training_args.do_predict:
        prediction_begin_time = time.time()
        trainer.model.set_predict_mode()
        trainer.prediction_loss_only = False
        trainer.data_collator.set_predict_mode()

        all_triples = (
            processor.get_train_triples(data_args.train_file)
            + processor.get_dev_triples()
            + processor.get_test_triples()
        )

        all_triples_set = {(t[0], t[1], t[2]) for t in all_triples}

        ranks, ranks_left, ranks_right = [], [], []
        hits = [[] for _ in range(10)]
        hits_left = [[] for _ in range(10)]
        hits_right = [[] for _ in range(10)]
        top_ten_hit_count = 0

        test_triples = processor.get_test_triples()
        total_test = len(test_triples)

        for test_id, test_triple in enumerate(test_triples):
            if np.random.random() > data_args.test_ratio:
                continue

            head, relation, tail = test_triple[0], test_triple[1], test_triple[2]

            # ---------------------------
            # 1. 预测头实体 (Predict Head)
            # ---------------------------
            head_corrupt_list = [test_triple]
            tmp_entity_list = (
                processor.rel2valid_head[relation]
                if data_args.type_constrain
                else processor.get_entities()
            )

            for corrupt_ent in tmp_entity_list:
                if corrupt_ent != head:
                    if (corrupt_ent, relation, tail) not in all_triples_set:
                        head_corrupt_list.append([corrupt_ent, relation, tail])

            _, tmp_features = processor._create_examples_and_features(head_corrupt_list)

            eval_data = DictDataset(
                input_ids=torch.tensor(
                    [f.input_ids for f in tmp_features], dtype=torch.long
                ),
                labels=torch.tensor(
                    [f.label_id for f in tmp_features], dtype=torch.long
                ),
                pos_indicator=torch.tensor(
                    [f.pos_indicator for f in tmp_features], dtype=torch.long
                ),
                head_neighbors=torch.stack(
                    [f.head_neighbors for f in tmp_features]
                ).long(),
                tail_neighbors=torch.stack(
                    [f.tail_neighbors for f in tmp_features]
                ).long(),
            )

            if hasattr(trainer.data_collator, "predict_mask_part"):
                trainer.data_collator.predict_mask_part = 0
            preds = trainer.predict(eval_data).predictions

            if trainer.is_world_master():
                argsort1 = np.argsort(-preds)
                rank1 = int(
                    np.where(argsort1 == 0)[0][0]
                )  # 找到真实三元组(索引0)的排名
                ranks.append(rank1 + 1)
                ranks_left.append(rank1 + 1)
                if rank1 < 10:
                    top_ten_hit_count += 1

            # ---------------------------
            # 2. 预测尾实体 (Predict Tail)
            # ---------------------------
            tail_corrupt_list = [test_triple]
            tmp_entity_list = (
                processor.rel2valid_tail[relation]
                if data_args.type_constrain
                else processor.get_entities()
            )

            for corrupt_ent in tmp_entity_list:
                if corrupt_ent != tail:
                    if (head, relation, corrupt_ent) not in all_triples_set:
                        tail_corrupt_list.append([head, relation, corrupt_ent])

            _, tmp_features = processor._create_examples_and_features(tail_corrupt_list)
            data_len = len(tmp_features)

            eval_data = DictDataset(
                input_ids=torch.tensor(
                    [f.input_ids for f in tmp_features], dtype=torch.long
                ),
                labels=torch.tensor(
                    [f.label_id for f in tmp_features], dtype=torch.long
                ),
                pos_indicator=torch.tensor(
                    [f.pos_indicator for f in tmp_features], dtype=torch.long
                ),
                head_neighbors=torch.stack(
                    [f.head_neighbors for f in tmp_features]
                ).long(),
                tail_neighbors=torch.stack(
                    [f.tail_neighbors for f in tmp_features]
                ).long(),
            )

            if hasattr(trainer.data_collator, "predict_mask_part"):
                trainer.data_collator.predict_mask_part = 2
            preds = trainer.predict(eval_data).predictions

            if trainer.is_world_master():
                argsort1 = np.argsort(-preds)
                rank2 = int(np.where(argsort1 == 0)[0][0])
                ranks.append(rank2 + 1)
                ranks_right.append(rank2 + 1)

                if rank2 < 10:
                    top_ten_hit_count += 1

                logger.info(
                    f"Test {test_id + 1}/{total_test} | L_Rank: {rank1 + 1} | R_Rank: {rank2 + 1} | Mean Rank: {np.mean(ranks):.2f} | Hits@10: {(top_ten_hit_count / len(ranks)):.4f}"
                )

                for hits_level in range(10):
                    hits_left[hits_level].append(1.0 if rank1 <= hits_level else 0.0)
                    hits_right[hits_level].append(1.0 if rank2 <= hits_level else 0.0)
                    hits[hits_level].append(1.0 if rank1 <= hits_level else 0.0)
                    hits[hits_level].append(1.0 if rank2 <= hits_level else 0.0)

        # ==========================
        # 打印最终评测指标
        # ==========================
        if trainer.is_world_master():
            logger.info("============= FINAL RESULTS =============")
            for i in [0, 2, 9]:  # Hits@1, Hits@3, Hits@10
                logger.info(f"Hits left   @{i + 1}: {np.mean(hits_left[i]):.4f}")
                logger.info(f"Hits right  @{i + 1}: {np.mean(hits_right[i]):.4f}")
                logger.info(f"Hits        @{i + 1}: {np.mean(hits[i]):.4f}")

            logger.info(f"Mean Rank left: {np.mean(ranks_left):.2f}")
            logger.info(f"Mean Rank right: {np.mean(ranks_right):.2f}")
            logger.info(f"Mean Rank:      {np.mean(ranks):.2f}")

            logger.info(f"MRR left:       {np.mean(1.0 / np.array(ranks_left)):.4f}")
            logger.info(f"MRR right:      {np.mean(1.0 / np.array(ranks_right)):.4f}")
            logger.info(f"MRR overall:    {np.mean(1.0 / np.array(ranks)):.4f}")
            logger.info(
                f"Total Time:     {(time.time() - prediction_begin_time) / 3600:.2f} Hours"
            )


if __name__ == "__main__":
    main()
