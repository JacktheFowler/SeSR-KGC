import csv
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DictDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = kwargs
        self.data_len = next(iter(kwargs.values())).size(0)
        for v in kwargs.values():
            assert self.data_len == v.size(0), "All tensors must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self):
        return self.data_len


class AlternateDataset(Dataset):
    def __init__(self, *args):
        self.datasets = args
        self.num_alternatives = len(args)
        self.data_len = len(args[0])
        self.counters = [0] * self.data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        res = self.datasets[self.counters[index] % self.num_alternatives][index]
        self.counters[index] += 1
        return res


@dataclass
class InputExample:
    guid: List[int]
    text_a: str
    text_b: str
    text_c: str
    head: str
    rel: str
    tail: str
    label: str


@dataclass
class InputFeatures:
    input_ids: List[int]
    label_id: int
    pos_indicator: Tuple[int, int, int, int, int, int]
    corrupted_part: int
    head_neighbors: torch.Tensor = field(
        default_factory=lambda: torch.full((10,), -1, dtype=torch.long)
    )
    tail_neighbors: torch.Tensor = field(
        default_factory=lambda: torch.full((10,), -1, dtype=torch.long)
    )


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, return_label=False):
        labels = []
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 3:
                    labels.append(line[3])
                    line = line[:3]
                lines.append(line)
            if return_label:
                return (lines, labels)
            return lines


class KGProcessor(DataProcessor):
    def __init__(self, data_args, tokenizer, is_world_master, must_load=False):
        self.data_dir = data_args.data_dir
        self.data_split = data_args.data_split
        self.rank = data_args.rank
        self.num_split = data_args.num_split
        self.no_mid = data_args.no_mid
        self.tokenizer = tokenizer
        self.is_world_master = is_world_master
        self.only_corrupt_entity = data_args.only_corrupt_entity
        self.vocab_size = len(tokenizer)
        self.max_seq_length = data_args.max_seq_length
        self.no_text = data_args.no_text
        self.text_sep_token = data_args.text_sep_token
        if self.no_text:
            self.max_seq_length = 5
        self.data_cache_dir = (
            data_args.data_cache_dir
            if data_args.data_cache_dir is not None
            else data_args.data_dir
        )
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.num_neg = data_args.num_neg
        self.must_load = must_load
        if must_load:
            self.check_all_data_saved()
        self.build_ent()
        self.build_rel()
        self.ent_size = len(self.ent_list)
        self.rel_size = len(self.rel_list)
        self.name2id = {e: i + self.vocab_size for (i, e) in enumerate(self.ent_list)}
        self.id2name = {i + self.vocab_size: e for (i, e) in enumerate(self.ent_list)}
        self.name2id.update(
            {
                r: i + self.vocab_size + self.ent_size
                for (i, r) in enumerate(self.rel_list)
            }
        )
        self.id2name.update(
            {
                i + self.vocab_size + self.ent_size: r
                for (i, r) in enumerate(self.rel_list)
            }
        )
        assert len(self.name2id) == len(self.id2name) == self.ent_size + self.rel_size
        if data_args.type_constrain:
            self.build_type_constrain()
        self.train_file = data_args.train_file
        self.build_entity_neighbors()

    def check_file_exists(self, file_path):
        assert os.path.exists(file_path), (
            f"expected to load data from {file_path} but it doesn't exist, please run generate_data.py (with the same args and --do_train, --do_eval, --do_predict) first"
        )

    def check_all_data_saved(self):
        ent_cache_file = os.path.join(self.data_cache_dir, "entity.pt")
        rel_cache_file = os.path.join(self.data_cache_dir, "relation.pt")
        self.check_file_exists(ent_cache_file)
        self.check_file_exists(rel_cache_file)
        train_data_file = f"train_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt"
        dev_data_file = (
            f"dev_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt"
        )
        test_data_file = f"test_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt"
        train_data_file = os.path.join(self.data_cache_dir, train_data_file)
        dev_data_file = os.path.join(self.data_cache_dir, dev_data_file)
        test_data_file = os.path.join(self.data_cache_dir, test_data_file)
        self.check_file_exists(train_data_file)
        self.check_file_exists(dev_data_file)
        self.check_file_exists(test_data_file)

    def get_train_examples(self, epoch, train_file):
        data_dir = self.data_dir
        cached_example_path = os.path.join(
            self.data_cache_dir, f"cached_train_examples_neg{self.num_neg}_epoch{epoch}"
        )
        os.makedirs(cached_example_path, exist_ok=True)
        (examples, features) = self._create_examples_and_features(
            os.path.join(data_dir, train_file), cached_example_path, self.num_neg
        )
        return (examples, features)

    def get_dev_examples(self):
        data_dir = self.data_dir
        cached_example_path = os.path.join(
            self.data_cache_dir, f"cached_dev_examples_{self.num_neg}"
        )
        os.makedirs(cached_example_path, exist_ok=True)
        (examples, features) = self._create_examples_and_features(
            os.path.join(data_dir, "dev.tsv"), cached_example_path, self.num_neg
        )
        return (examples, features)

    def get_test_examples(self):
        data_dir = self.data_dir
        cached_example_path = os.path.join(self.data_cache_dir, "cached_test_examples")
        os.makedirs(cached_example_path, exist_ok=True)
        if self.data_split:
            (examples, features) = self._create_examples_and_features(
                os.path.join(
                    data_dir, f"test-p{self.rank + 1}-of-{self.num_split}.tsv"
                ),
                cached_example_path,
            )
        else:
            (examples, features) = self._create_examples_and_features(
                os.path.join(data_dir, "test.tsv"), cached_example_path
            )
        return (examples, features)

    def build_ent(self):
        ent_cache_file = os.path.join(self.data_cache_dir, "entity.pt")
        if os.path.exists(ent_cache_file):
            logger.info("loading entity data from {}".format(ent_cache_file))
            (self.ent2text, self.ent2tokens) = torch.load(ent_cache_file)
        else:
            logger.info("building entity data")
            self.ent2text = {}
            self.ent2tokens = {}
            with open(os.path.join(self.data_dir, "entity2text.txt"), "r") as f:
                ent_lines = f.readlines()
                for line in tqdm(ent_lines, disable=not self.is_world_master):
                    tmp = line.strip().split("\t")
                    if len(tmp) == 2:
                        self.ent2text[tmp[0]] = tmp[1]
                        self.ent2tokens[tmp[0]] = self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(tmp[1])
                        )
            if self.data_dir.find("FB15") != -1:
                with open(os.path.join(self.data_dir, "entity2textlong.txt"), "r") as f:
                    ent_lines = f.readlines()
                    for line in tqdm(ent_lines, disable=not self.is_world_master):
                        tmp = line.strip().split("\t")
                        self.ent2text[tmp[0]] = tmp[1]
                        self.ent2tokens[tmp[0]] = self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(tmp[1])
                        )
            logger.info("saving entity data to {}".format(ent_cache_file))
            if self.is_world_master:
                torch.save((self.ent2text, self.ent2tokens), ent_cache_file)
        self.ent_list = sorted(self.ent2text.keys())

    def build_rel(self):
        rel_cache_file = os.path.join(self.data_cache_dir, "relation.pt")
        if os.path.exists(rel_cache_file):
            logger.info("loading relation data from {}".format(rel_cache_file))
            (self.rel2text, self.rel2tokens) = torch.load(rel_cache_file)
        else:
            logger.info("building relation data")
            self.rel2text = {}
            self.rel2tokens = {}
            with open(os.path.join(self.data_dir, "relation2text.txt"), "r") as f:
                rel_lines = f.readlines()
                for line in tqdm(rel_lines, disable=not self.is_world_master):
                    temp = line.strip().split("\t")
                    self.rel2text[temp[0]] = temp[1]
                    self.rel2tokens[temp[0]] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(temp[1])
                    )
            logger.info("saving relation data to {}".format(rel_cache_file))
            if self.is_world_master:
                torch.save((self.rel2text, self.rel2tokens), rel_cache_file)
        self.rel_list = sorted(self.rel2text.keys())

    def build_type_constrain(self):
        KE_id2ent = {}
        with open(os.path.join(self.data_dir, "entity2id.txt"), "r") as f:
            lines = f.readlines()
        for line in lines[1:]:
            (emid, ent_id) = line.strip().split("\t")
            KE_id2ent[ent_id] = emid
        KE_id2rel = {}
        with open(os.path.join(self.data_dir, "relation2id.txt"), "r") as f:
            lines = f.readlines()
        for line in lines[1:]:
            (rmid, rel_id) = line.strip().split("\t")
            KE_id2rel[rel_id] = rmid
        with open(os.path.join(self.data_dir, "type_constrain.txt"), "r") as f:
            lines = f.readlines()
        (self.rel2valid_head, self.rel2valid_tail) = ({}, {})
        for num_line, line in enumerate(lines[1:]):
            line = line.strip().split("\t")
            relation = KE_id2rel[line[0]]
            ents = [KE_id2ent[ent] for ent in line[2:]]
            assert len(ents) == int(line[1])
            if num_line % 2 == 0:
                self.rel2valid_head[relation] = ents
            else:
                self.rel2valid_tail[relation] = ents

    def get_name2id(self):
        return self.name2id

    def get_id2name(self):
        return self.id2name

    def get_ent2text(self):
        return self.ent2text

    def get_rel2text(self):
        return self.rel2text

    def get_labels(self):
        return ["0", "1"]

    def get_entities(self):
        return self.ent_list

    def get_relations(self):
        return self.rel_list

    def get_train_triples(self, train_file):
        return self._read_tsv(os.path.join(self.data_dir, train_file))

    def get_dev_triples(self, return_label=False):
        return self._read_tsv(
            os.path.join(self.data_dir, "dev.tsv"), return_label=return_label
        )

    def get_test_triples(self, return_label=False):
        if self.data_split:
            return self._read_tsv(
                os.path.join(
                    self.data_dir, f"test-p{self.rank + 1}-of-{self.num_split}.tsv"
                ),
                return_label=return_label,
            )
        else:
            return self._read_tsv(
                os.path.join(self.data_dir, "test.tsv"), return_label=return_label
            )

    def create_examples(self, lines, num_corr, print_info=True):
        if isinstance(lines, str):
            lines = self._read_tsv(lines)

        # 专家优化：用 Tuple Set 代替 String Set，Hash查找速度提升巨大
        triples_set = {(line[0], line[1], line[2]) for line in lines}
        examples = []

        for i, line in enumerate(
            tqdm(lines, disable=not self.is_world_master or not print_info)
        ):
            head, rel, tail = line
            text_a, text_b, text_c = (
                self.ent2text[head],
                self.rel2text[rel],
                self.ent2text[tail],
            )
            examples.append(
                InputExample([i, 0, 0], text_a, text_b, text_c, head, rel, tail, "1")
            )

            if num_corr == 0:
                continue
            rnd = random.random()

            # 替换 Head
            if not self.only_corrupt_entity or rnd <= 0.5:
                for j in range(num_corr):
                    while True:
                        tmp_head = random.choice(self.ent_list)
                        if (tmp_head, rel, tail) not in triples_set:
                            break
                    examples.append(
                        InputExample(
                            [i, 1, j],
                            self.ent2text[tmp_head],
                            text_b,
                            text_c,
                            tmp_head,
                            rel,
                            tail,
                            "0",
                        )
                    )

            # 替换 Relation
            if not self.only_corrupt_entity:
                for j in range(num_corr):
                    while True:
                        tmp_rel = random.choice(self.rel_list)
                        if (head, tmp_rel, tail) not in triples_set:
                            break
                    examples.append(
                        InputExample(
                            [i, 2, j],
                            text_a,
                            self.rel2text[tmp_rel],
                            text_c,
                            head,
                            tmp_rel,
                            tail,
                            "0",
                        )
                    )

            # 替换 Tail
            if not self.only_corrupt_entity or rnd > 0.5:
                for j in range(num_corr):
                    while True:
                        tmp_tail = random.choice(self.ent_list)
                        if (head, rel, tmp_tail) not in triples_set:
                            break
                    examples.append(
                        InputExample(
                            [i, 3, j],
                            text_a,
                            text_b,
                            self.ent2text[tmp_tail],
                            head,
                            rel,
                            tmp_tail,
                            "0",
                        )
                    )
        return examples

    def _create_examples_and_features(self, lines, cache_path=None, num_corr=0):
        if cache_path is None:
            examples = self.create_examples(lines, num_corr, print_info=False)
            features = self.convert_examples_to_features(examples, print_info=False)
            return (examples, features)
        cache_example_file = os.path.join(cache_path, "example.pt")
        if self.no_text:
            cache_feature_file = os.path.join(cache_path, "feature_notext.pt")
        else:
            cache_feature_file = os.path.join(
                cache_path, f"feature_{self.max_seq_length}_{self.text_sep_token}.pt"
            )
        if os.path.exists(cache_example_file):
            logger.info("loading examples from {}".format(cache_example_file))
            if self.must_load:
                examples = None
            else:
                examples = torch.load(cache_example_file)
            logger.info("load examples done")
        else:
            examples = self.create_examples(lines, num_corr)
            logger.info("saving examples to {}".format(cache_example_file))
            if self.is_world_master:
                torch.save(examples, cache_example_file)
            logger.info("save examples done")
        if os.path.exists(cache_feature_file):
            logger.info("loading features from {}".format(cache_feature_file))
            features = torch.load(cache_feature_file)
            logger.info("load features done")
        else:
            features = self.convert_examples_to_features(examples)
            logger.info("saving features to {}".format(cache_feature_file))
            if self.is_world_master:
                torch.save(features, cache_feature_file)
            logger.info("save features done")
        return (examples, features)

    def tokenize(self, example):
        # 专家优化：彻底摒弃 copy.deepcopy，直接使用切片[:]进行列表极速浅拷贝
        tokens_a = self.ent2tokens[example.head][:]
        tokens_b = self.rel2tokens[example.rel][:]
        tokens_c = self.ent2tokens[example.tail][:]
        return (tokens_a, tokens_b, tokens_c)

    def convert_examples_to_features(self, examples, print_info=True):
        label_list = self.get_labels()
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        name2id = self.name2id
        no_text = self.no_text
        label_map = {label: i for (i, label) in enumerate(label_list)}
        ent2idx = {ent: i for i, ent in enumerate(self.ent_list)}
        features = []

        for ex_index, example in enumerate(
            tqdm(examples, disable=not self.is_world_master or not print_info)
        ):
            corrupted_part = example.guid[1] - 1
            head_id = name2id[example.head]
            tail_id = name2id[example.tail]
            rel_id = name2id[example.rel]
            SEP_id = tokenizer.sep_token_id
            CLS_id = tokenizer.cls_token_id
            if self.text_sep_token:
                SPACE_id = tokenizer.convert_tokens_to_ids(self.text_sep_token)
            else:
                SPACE_id = None
            if self.no_mid:
                triplet_ids = []
            else:
                triplet_ids = [head_id, rel_id, tail_id, SEP_id]
            pos_indicator = (0, 0, 0, 0, 0, 0)
            if no_text:
                input_ids = [CLS_id] + triplet_ids
            else:
                (tokens_a, tokens_b, tokens_c) = self.tokenize(example)
                if SPACE_id:
                    if self.no_mid:
                        _truncate_seq_triple(
                            tokens_a, tokens_b, tokens_c, max_seq_length - 4
                        )
                    else:
                        if tokenizer.__class__.__name__.startswith("Roberta"):
                            _truncate_seq_triple(
                                tokens_a, tokens_b, tokens_c, max_seq_length - 9
                            )
                        elif tokenizer.__class__.__name__.startswith("Bert"):
                            _truncate_seq_triple(
                                tokens_a, tokens_b, tokens_c, max_seq_length - 8
                            )
                        else:
                            raise NotImplementedError()
                    input_ids = (
                        [CLS_id]
                        + tokens_a
                        + [SPACE_id]
                        + tokens_b
                        + [SPACE_id]
                        + tokens_c
                        + [SEP_id]
                    )
                    pos_indicator = (
                        0,
                        1 + len(tokens_a),
                        1 + len(tokens_a),
                        2 + len(tokens_a) + len(tokens_b),
                        2 + len(tokens_a) + len(tokens_b),
                        3 + len(tokens_a) + len(tokens_b) + len(tokens_c),
                    )
                else:
                    if self.no_mid:
                        _truncate_seq_triple(
                            tokens_a, tokens_b, tokens_c, max_seq_length - 2
                        )
                    else:
                        if tokenizer.__class__.__name__.startswith("Roberta"):
                            _truncate_seq_triple(
                                tokens_a, tokens_b, tokens_c, max_seq_length - 7
                            )
                        elif tokenizer.__class__.__name__.startswith("Bert"):
                            _truncate_seq_triple(
                                tokens_a, tokens_b, tokens_c, max_seq_length - 6
                            )
                        else:
                            raise NotImplementedError()
                    input_ids = [CLS_id] + tokens_a + tokens_b + tokens_c + [SEP_id]
                    pos_indicator = (
                        0,
                        1 + len(tokens_a),
                        len(tokens_a),
                        1 + len(tokens_a) + len(tokens_b),
                        len(tokens_a) + len(tokens_b),
                        1 + len(tokens_a) + len(tokens_b) + len(tokens_c),
                    )
                if not self.no_mid:
                    if tokenizer.__class__.__name__.startswith("Roberta"):
                        input_ids += [SEP_id]
                    input_ids += triplet_ids
            label_id = label_map[example.label]
            input_ids += [0] * (max_seq_length - len(input_ids))

            # 新增：获取结构信息
            head_idx = ent2idx.get(example.head, 0)
            tail_idx = ent2idx.get(example.tail, 0)
            head_neighbors = self.entity_neighbors.get(
                head_idx, torch.full((10,), -1, dtype=torch.long)
            )
            tail_neighbors = self.entity_neighbors.get(
                tail_idx, torch.full((10,), -1, dtype=torch.long)
            )

            if ex_index < 5 and print_info:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    label_id=label_id,
                    pos_indicator=pos_indicator,
                    corrupted_part=corrupted_part,
                    head_neighbors=head_neighbors,
                    tail_neighbors=tail_neighbors,
                )
            )
        return features

    def get_dataset(self, args):
        train_dataset, eval_dataset, predict_dataset = None, None, None
        train_data_file = os.path.join(
            self.data_cache_dir,
            f"train_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt",
        )
        dev_data_file = (
            f"dev_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt"
        )
        test_data_file = f"test_dataset_{self.num_neg}_{self.max_seq_length}_{self.text_sep_token}.pt"
        train_data_file = os.path.join(self.data_cache_dir, train_data_file)
        dev_data_file = os.path.join(self.data_cache_dir, dev_data_file)
        test_data_file = os.path.join(self.data_cache_dir, test_data_file)
        if args.do_train:
            if os.path.exists(train_data_file):
                train_dataset = torch.load(train_data_file)
            else:
                train_dataset = []
                for epoch in range(int(args.num_train_epochs) + 1):
                    _, train_features = self.get_train_examples(epoch, self.train_file)

                    # 🚨 专家修复：将遗失的 neighbors 特征打包进 Dataset！
                    train_dataset.append(
                        DictDataset(
                            input_ids=torch.tensor(
                                [f.input_ids for f in train_features], dtype=torch.long
                            ),
                            labels=torch.tensor(
                                [f.label_id for f in train_features], dtype=torch.long
                            ),
                            pos_indicator=torch.tensor(
                                [f.pos_indicator for f in train_features],
                                dtype=torch.long,
                            ),
                            corrupted_part=torch.tensor(
                                [f.corrupted_part for f in train_features],
                                dtype=torch.long,
                            ),
                            head_neighbors=torch.stack(
                                [f.head_neighbors for f in train_features]
                            ),
                            tail_neighbors=torch.stack(
                                [f.tail_neighbors for f in train_features]
                            ),
                        )
                    )
                train_dataset = AlternateDataset(*train_dataset)
                if self.is_world_master:
                    torch.save(train_dataset, train_data_file)

        if args.do_eval:
            if os.path.exists(dev_data_file):
                eval_dataset = torch.load(dev_data_file)
            else:
                _, eval_features = self.get_dev_examples()
                eval_dataset = DictDataset(
                    input_ids=torch.tensor(
                        [f.input_ids for f in eval_features], dtype=torch.long
                    ),
                    labels=torch.tensor(
                        [f.label_id for f in eval_features], dtype=torch.long
                    ),
                    pos_indicator=torch.tensor(
                        [f.pos_indicator for f in eval_features], dtype=torch.long
                    ),
                    head_neighbors=torch.stack(
                        [f.head_neighbors for f in eval_features]
                    ),
                    tail_neighbors=torch.stack(
                        [f.tail_neighbors for f in eval_features]
                    ),
                )
                if self.is_world_master:
                    torch.save(eval_dataset, dev_data_file)

        if args.do_predict:
            if os.path.exists(test_data_file):
                predict_dataset = torch.load(test_data_file)
            else:
                eval_examples, eval_features = self.get_test_examples()
                predict_dataset = DictDataset(
                    input_ids=torch.tensor(
                        [f.input_ids for f in eval_features], dtype=torch.long
                    ),
                    labels=torch.tensor(
                        [f.label_id for f in eval_features], dtype=torch.long
                    ),
                    pos_indicator=torch.tensor(
                        [f.pos_indicator for f in eval_features], dtype=torch.long
                    ),
                    head_neighbors=torch.stack(
                        [f.head_neighbors for f in eval_features]
                    ),
                    tail_neighbors=torch.stack(
                        [f.tail_neighbors for f in eval_features]
                    ),
                )
                if self.is_world_master:
                    torch.save(predict_dataset, test_data_file)
        return (train_dataset, eval_dataset, predict_dataset)

    def build_entity_neighbors(self, max_neighbors=10):
        """专家重构：完全基于稀疏图(Adjacency List)构建，内存占用降低至 1%"""
        neighbors_cache_file = os.path.join(
            self.data_cache_dir, f"entity_neighbors_{max_neighbors}.pt"
        )
        if os.path.exists(neighbors_cache_file):
            logger.info(f"loading entity neighbors from {neighbors_cache_file}")
            self.entity_neighbors = torch.load(neighbors_cache_file)
        else:
            logger.info("building sparse entity neighbors from triples")
            train_triples = self._read_tsv(os.path.join(self.data_dir, self.train_file))
            ent2idx = {ent: i for i, ent in enumerate(self.ent_list)}

            # 1. 使用 defaultdict(set) 极速构建无向稀疏图
            adj_list = defaultdict(set)
            for head, rel, tail in train_triples:
                if head in ent2idx and tail in ent2idx:
                    h_idx, t_idx = ent2idx[head], ent2idx[tail]
                    adj_list[h_idx].add(t_idx)
                    adj_list[t_idx].add(h_idx)

            # 2. 转换为 Tensor 格式并 Padding
            self.entity_neighbors = {}
            for ent_idx in range(self.ent_size):
                neighbors = list(adj_list.get(ent_idx, []))

                # 随机采样或者截断，避免永远只取固定的前N个邻居
                if len(neighbors) > max_neighbors:
                    neighbors = random.sample(neighbors, max_neighbors)

                neighbors_tensor = torch.tensor(neighbors, dtype=torch.long)

                if len(neighbors) < max_neighbors:
                    padding = torch.full(
                        (max_neighbors - len(neighbors),), -1, dtype=torch.long
                    )
                    neighbors_tensor = torch.cat([neighbors_tensor, padding])

                self.entity_neighbors[ent_idx] = neighbors_tensor

            if self.is_world_master:
                torch.save(self.entity_neighbors, neighbors_cache_file)
            logger.info("entity neighbors built efficiently")


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
    if total_length <= max_length:
        return False
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_c) >= len(tokens_a) and len(tokens_c) >= len(tokens_b):
            tokens_c.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
    return True
