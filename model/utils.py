import contextlib
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    if seed is None:
        yield
        return

    if len(addl_seeds) > 0:
        # 使用确定性的字符串拼接和 MD5 哈希
        seed_str = f"{seed}_" + "_".join(str(s) for s in addl_seeds)
        md5_hash = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
        # 取前 7 个十六进制字符（确保在 32 位整数范围内，且足够随机）
        seed = int(md5_hash[:7], 16)

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    pooling_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to use mean pooling of text encoding for triplet modeling"
        },
    )
    text_loss_weight: float = field(
        default=0.1, metadata={"help": "The weight of text loss (MLM)"}
    )
    pos_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": "The weight of positive labels in BCE loss. Should balance the negative sampling ratio."
        },
    )

    # 结构信息与融合相关超参数
    structure_loss_weight: float = field(
        default=0.3, metadata={"help": "The weight of InfoNCE structure loss"}
    )
    reconstruction_loss_weight: float = field(
        default=0.2, metadata={"help": "The weight of auto-encoder reconstruction loss"}
    )
    gating_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for adaptive gating fusion module"},
    )
    # 专家修复：解决命名冲突，改名为 gnn_mask_ratio
    gnn_mask_ratio: float = field(
        default=0.15,
        metadata={"help": "The ratio of masked structural edges/entities in GNN"},
    )
    # 专家新增：对比学习的关键超参数
    temperature: float = field(
        default=0.07,
        metadata={
            "help": "Temperature hyper-parameter for structural contrastive InfoNCE loss."
        },
    )


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The input data dir. Should contain .tsv files."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the examples and features"},
    )
    num_neg: int = field(
        default=1,
        metadata={"help": "The number of negative samples per positive triplet."},
    )
    margin: float = field(
        default=1.0,
        metadata={"help": "The margin of distance-based knowledge loss (e.g., TransE)"},
    )
    no_text: bool = field(
        default=False,
        metadata={
            "help": "Whether to completely drop text input and run purely on graph"
        },
    )
    data_debug: bool = field(
        default=False,
        metadata={
            "help": "Whether to use only a small subset of data for fast debugging"
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after WordPiece tokenization."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens."}
    )

    # 专家修复：解决命名冲突，改名为 mlm_mask_ratio (与 DataCollator 的 mlm_probability 对应)
    mlm_mask_ratio: float = field(
        default=0.15,
        metadata={"help": "The ratio of text tokens to be masked in MLM task"},
    )

    # 专家新增：防 OOM 与图感受野控制的超级核心参数
    max_neighbors: int = field(
        default=10,
        metadata={
            "help": "Maximum number of topological neighbors to retain per entity."
        },
    )

    group_shuffle: bool = field(
        default=False,
        metadata={
            "help": "Whether to use group shuffle (pos and neg samples in the same batch)"
        },
    )
    test_ratio: float = field(
        default=1.0, metadata={"help": "The ratio of test data used to evaluate"}
    )
    text_sep_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The special token added between head, relation, and tail text."
        },
    )
    type_constrain: bool = field(
        default=False,
        metadata={
            "help": "Whether to use entity type constraints during negative sampling (Boosts MRR!)"
        },
    )
    no_mid: bool = field(default=False)
    data_split: bool = field(default=False)
    num_split: int = field(default=5)
    rank: int = field(default=0)
    only_corrupt_entity: bool = field(
        default=False,
        metadata={
            "help": "If true, only generate corrupted heads/tails, not relations."
        },
    )
    train_file: str = field(default="train.tsv")
