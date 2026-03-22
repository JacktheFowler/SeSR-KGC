import torch


class PoolingCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.predict_mode = False
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def process_label(self, batch_data):
        if self.predict_mode:
            keys_to_remove = [
                k
                for k in [
                    "labels",
                    "lm_labels",
                    "masked_lm_labels",
                    "mlm_labels",
                    "span_labels",
                ]
                if k in batch_data
            ]
            for k in keys_to_remove:
                batch_data.pop(k)
        return batch_data

    def span_mask(self, inputs, mask_left=None, mask_right=None):
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer without mask token.")

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if mask_left is not None and mask_right is not None:
            probability_matrix.fill_(0.0)
            # 切片操作前确保不越界
            probability_matrix[:, mask_left:mask_right] = self.mlm_probability

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )

        # 专家修复：移除冗余的 torch.tensor 包裹，消除 Warning
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        out_of_word_mask = labels >= len(self.tokenizer)
        probability_matrix.masked_fill_(out_of_word_mask, value=0.0)

        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def prepare_structure_info(self, examples):
        if "head_neighbors" not in examples[0]:
            return {}

        head_neighbors = torch.stack([ex["head_neighbors"] for ex in examples])
        tail_neighbors = torch.stack([ex["tail_neighbors"] for ex in examples])

        head_neighbor_mask = (head_neighbors != -1).long()
        tail_neighbor_mask = (tail_neighbors != -1).long()

        head_neighbors = head_neighbors.clamp(min=0)
        tail_neighbors = tail_neighbors.clamp(min=0)

        return {
            "head_neighbors": head_neighbors,
            "tail_neighbors": tail_neighbors,
            "head_neighbor_mask": head_neighbor_mask,
            "tail_neighbor_mask": tail_neighbor_mask,
        }

    def __call__(self, examples):
        max_seq_length = examples[0]["input_ids"].size(0)
        batch_size = len(examples)

        # 专家修复：预分配批次级别的张量矩阵，彻底消灭 For 循环里的列表拼接！
        attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        pooling_head_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        pooling_rel_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        pooling_tail_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        for i, example in enumerate(examples):
            assert example["input_ids"].size(0) == max_seq_length, (
                "Input lengths must match."
            )

            pos = example["pos_indicator"]

            # 使用切片直接赋值，告别 [0]*N + [1]*M + [0]*K
            attention_mask[i, : pos[-1].item() + 1] = 1
            pooling_head_mask[i, 1 : pos[1].item()] = 1
            pooling_rel_mask[i, pos[2].item() + 1 : pos[3].item()] = 1
            pooling_tail_mask[i, pos[4].item() + 1 : pos[5].item()] = 1

        # 排除不需要直接 stack 的原始控制字段
        exclude_keys = {
            "pos_indicator",
            "corrupted_part",
            "head_neighbors",
            "tail_neighbors",
        }
        input_keys = [k for k in examples[0].keys() if k not in exclude_keys]

        # 优雅地构建 batch_data
        batch_data = {k: torch.stack([ex[k] for ex in examples]) for k in input_keys}

        # 将我们极速生成的掩码并入 batch_data，保护原始 examples 字典不被污染
        batch_data.update(
            {
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "pooling_head_mask": pooling_head_mask,
                "pooling_rel_mask": pooling_rel_mask,
                "pooling_tail_mask": pooling_tail_mask,
            }
        )

        # 追加结构化图信息
        batch_data.update(self.prepare_structure_info(examples))

        # 【可选提醒】：如果你在使用 MLM，请在这里对 batch_data["input_ids"] 调一次 self.span_mask
        # 例如：
        # if not self.predict_mode and self.mlm_probability > 0:
        #     inputs, mlm_labels = self.span_mask(batch_data["input_ids"])
        #     batch_data["input_ids"] = inputs
        #     batch_data["mlm_labels"] = mlm_labels

        return self.process_label(batch_data)
