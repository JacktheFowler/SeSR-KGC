import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel


class GatingFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, head, rel, tail):
        fused = torch.cat([head, rel, tail], dim=-1)
        return self.gate(fused).squeeze(-1)


class StructureReconstructor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head_reconstructor = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tail_reconstructor = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, head, rel, tail):
        head_recon = self.head_reconstructor(torch.cat([rel, tail], dim=-1))
        tail_recon = self.tail_reconstructor(torch.cat([head, rel], dim=-1))

        head_loss = F.mse_loss(head_recon, head.detach())
        tail_loss = F.mse_loss(tail_recon, tail.detach())

        return (head_loss + tail_loss) * 0.5


class PoolingTripletDecoder(nn.Module):
    def __init__(self, margin, hidden_dim=None, use_structure=False):
        super().__init__()
        self.margin = margin
        self.use_structure = use_structure

        if use_structure and hidden_dim is not None:
            self.gating = GatingFusion(hidden_dim)
            self.reconstructor = StructureReconstructor(hidden_dim)
            self.structure_proj = nn.Linear(hidden_dim, hidden_dim)
            self.temp = 0.07

    def compute_structure_loss(
        self,
        head,
        rel,
        tail,
        head_neighbors,
        tail_neighbors,
        head_neighbor_mask,
        tail_neighbor_mask,
        entity_embeddings,
    ):
        batch_size, max_neighbors = head_neighbors.size()

        head_neighbor_emb = entity_embeddings[head_neighbors]
        tail_neighbor_emb = entity_embeddings[tail_neighbors]
        head_neighbor_emb = F.normalize(head_neighbor_emb, p=2, dim=-1)
        tail_neighbor_emb = F.normalize(tail_neighbor_emb, p=2, dim=-1)

        head_proj = F.normalize(self.structure_proj(head), p=2, dim=-1).unsqueeze(1)
        tail_proj = F.normalize(self.structure_proj(tail), p=2, dim=-1).unsqueeze(1)

        head_sim = (
            torch.bmm(head_proj, head_neighbor_emb.transpose(1, 2)).squeeze(1)
            / self.temp
        )
        tail_sim = (
            torch.bmm(tail_proj, tail_neighbor_emb.transpose(1, 2)).squeeze(1)
            / self.temp
        )

        head_sim = head_sim.masked_fill(head_neighbor_mask == 0, -1e9)
        tail_sim = tail_sim.masked_fill(tail_neighbor_mask == 0, -1e9)

        head_loss = -F.log_softmax(head_sim, dim=-1).mean()
        tail_loss = -F.log_softmax(tail_sim, dim=-1).mean()

        return (head_loss + tail_loss) * 0.5

    def forward(self, encoding_triplet, structure_info=None, entity_embeddings=None):
        head, rel, tail = (
            encoding_triplet[:, 0, :],
            encoding_triplet[:, 1, :],
            encoding_triplet[:, 2, :],
        )

        distance = ((head + rel - tail) ** 2).sum(dim=-1) / (
            encoding_triplet.size(-1) ** 0.5
        )
        z = self.margin - 0.5 * distance

        if not self.use_structure or structure_info is None:
            return z, None, None, None

        struct_loss = self.compute_structure_loss(
            head,
            rel,
            tail,
            structure_info["head_neighbors"],
            structure_info["tail_neighbors"],
            structure_info["head_neighbor_mask"],
            structure_info["tail_neighbor_mask"],
            entity_embeddings,
        )

        recon_loss = self.reconstructor(head, rel, tail)
        gate = self.gating(head, rel, tail)
        gated_z = self.margin - 0.5 * (distance * (1.0 - gate * 0.5))

        return gated_z, struct_loss, recon_loss, gate


class BertPoolingForTripletPrediction(BertPreTrainedModel):
    def __init__(
        self,
        config,
        margin,
        text_loss_weight,
        pos_weight,
        use_structure=False,
        num_entities=0,
    ):
        super().__init__(config)
        self.bert = BertModel(config)
        self.use_structure = use_structure
        hidden_dim = config.hidden_size

        self.trip_decoder = PoolingTripletDecoder(
            margin,
            hidden_dim=hidden_dim if use_structure else None,
            use_structure=use_structure,
        )

        if use_structure and num_entities > 0:
            self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
            nn.init.xavier_uniform_(self.entity_embeddings.weight)

        embed_size = config.vocab_size
        config.vocab_size = getattr(config, "real_vocab_size", embed_size)
        self.cls = BertOnlyMLMHead(config)

        self.predict_mode = False
        self.text_loss_weight = text_loss_weight

        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight_tensor", pos_weight)
        self.trip_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        self.lm_loss_fct = nn.CrossEntropyLoss()

        config.vocab_size = embed_size
        self.init_weights()

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def pooling_encoder(self, sequence_output, pooling_mask):
        pooling_mask = pooling_mask.type_as(sequence_output)
        mask_len = torch.sum(pooling_mask, dim=1, keepdim=True).clamp(min=1e-9)
        mask_sum = torch.matmul(pooling_mask.unsqueeze(1), sequence_output).squeeze(1)
        return mask_sum / mask_len

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pooling_head_mask=None,
        pooling_rel_mask=None,
        pooling_tail_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mlm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        head_neighbors=None,
        tail_neighbors=None,
        head_neighbor_mask=None,
        tail_neighbor_mask=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        head_encoding = self.pooling_encoder(sequence_output, pooling_head_mask)
        rel_encoding = self.pooling_encoder(sequence_output, pooling_rel_mask)
        tail_encoding = self.pooling_encoder(sequence_output, pooling_tail_mask)
        trip_encoding = torch.stack([head_encoding, rel_encoding, tail_encoding], dim=1)

        structure_info, entity_emb = None, None
        if self.use_structure and head_neighbors is not None:
            structure_info = {
                "head_neighbors": head_neighbors,
                "tail_neighbors": tail_neighbors,
                "head_neighbor_mask": head_neighbor_mask,
                "tail_neighbor_mask": tail_neighbor_mask,
            }
            entity_emb = self.entity_embeddings.weight

        triplet_scores, struct_loss, recon_loss, gate = self.trip_decoder(
            trip_encoding, structure_info, entity_emb
        )

        if self.predict_mode:
            return (triplet_scores,)

        total_loss = None
        if labels is not None:
            total_loss = self.trip_loss_fct(triplet_scores, labels.float())

        if mlm_labels is not None and prediction_scores is not None:
            masked_lm_loss = self.text_loss_weight * self.lm_loss_fct(
                prediction_scores.view(-1, self.config.real_vocab_size),
                mlm_labels.view(-1),
            )
            total_loss = (
                masked_lm_loss if total_loss is None else total_loss + masked_lm_loss
            )

        if struct_loss is not None:
            alpha, beta = 1.0, 1.0
            total_loss = total_loss + alpha * struct_loss + beta * recon_loss

        return (
            (total_loss, triplet_scores)
            if total_loss is not None
            else (triplet_scores,)
        )
