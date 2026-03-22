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
            probability_matrix[:, mask_left:mask_right] = self.mlm_probability

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
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

            attention_mask[i, : pos[-1].item() + 1] = 1
            pooling_head_mask[i, 1 : pos[1].item()] = 1
            pooling_rel_mask[i, pos[2].item() + 1 : pos[3].item()] = 1
            pooling_tail_mask[i, pos[4].item() + 1 : pos[5].item()] = 1

        exclude_keys = {
            "pos_indicator",
            "corrupted_part",
            "head_neighbors",
            "tail_neighbors",
        }
        input_keys = [k for k in examples[0].keys() if k not in exclude_keys]

        batch_data = {k: torch.stack([ex[k] for ex in examples]) for k in input_keys}
        batch_data.update(
            {
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "pooling_head_mask": pooling_head_mask,
                "pooling_rel_mask": pooling_rel_mask,
                "pooling_tail_mask": pooling_tail_mask,
            }
        )

        batch_data.update(self.prepare_structure_info(examples))

        if not self.predict_mode and self.mlm_probability > 0:
            inputs, mlm_labels = self.span_mask(batch_data["input_ids"])
            batch_data["input_ids"] = inputs
            batch_data["mlm_labels"] = mlm_labels

        return self.process_label(batch_data)
