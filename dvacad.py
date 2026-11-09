import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
except ImportError as exc:
    raise ImportError("Qwen3-VL requires transformers>=4.57.0") from exc


DEFAULT_VISUAL_MEMORY_MAX_SPANS = 0


class SequenceEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers=3, num_heads=8, intermediate_size=None):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, intermediate_size or 4 * hidden_size,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.gradient_checkpointing = False

    def forward(self, embeddings, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        if not attention_mask.any(dim=1).all():
            raise ValueError("SequenceEncoder requires a non-empty sequence per sample")
        # Positions count valid tokens, so left padding does not change a sequence.
        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp_min(0).float()
        frequencies = torch.exp(
            torch.arange(0, embeddings.shape[-1], 2, device=embeddings.device).float()
            * (-math.log(10000.0) / embeddings.shape[-1])
        )
        angles = positions.unsqueeze(-1) * frequencies
        positional = torch.zeros_like(embeddings)
        positional[..., 0::2] = angles.sin().to(embeddings.dtype)
        positional[..., 1::2] = angles[..., :positional[..., 1::2].shape[-1]].cos().to(embeddings.dtype)
        hidden = embeddings + positional
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            for layer in self.layers.layers:
                hidden = checkpoint(layer, hidden, src_key_padding_mask=~attention_mask, use_reentrant=False)
        else:
            hidden = self.layers(hidden, src_key_padding_mask=~attention_mask)
        return hidden.masked_fill(~attention_mask.unsqueeze(-1), 0)


class MatchingProjection(nn.Sequential):
    def __init__(self, hidden_size):
        super().__init__(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))


def cosine_cost(sequence_features, visual_features, eps=1e-8):
    sequence_features = F.normalize(sequence_features.float(), dim=-1, eps=eps)
    visual_features = F.normalize(visual_features.float(), dim=-1, eps=eps)
    return 1.0 - sequence_features @ visual_features.transpose(-1, -2)


def unbalanced_sinkhorn(cost, dustbin_cost, epsilon=0.1, rho_sequence=1.0,
                        rho_visual=1.0, dustbin_mass=0.1, n_iters=80):
    """Eq. (3), including KL(T || a b^T), solved in the log domain.

    Real entries have uniform mass 1/N; the dustbin has additional mass 0.1.
    Return the full, *unnormalized* transport including both dustbins.
    """
    if epsilon <= 0 or rho_sequence <= 0 or rho_visual <= 0 or dustbin_mass <= 0 or n_iters <= 0:
        raise ValueError("Sinkhorn regularization, prior masses and iterations must be positive")
    cost = cost.float()
    if cost.ndim != 3 or min(cost.shape[-2:]) == 0 or not torch.isfinite(cost).all():
        raise ValueError("Expected a finite non-empty [batch, sequence, visual] cost matrix")
    batch_size, ns, nv = cost.shape
    dustbin = torch.as_tensor(dustbin_cost, device=cost.device, dtype=cost.dtype)
    upper = torch.cat([cost, dustbin.expand(batch_size, ns, 1)], dim=-1)
    lower = torch.cat([dustbin.expand(batch_size, 1, nv), cost.new_zeros(batch_size, 1, 1)], dim=-1)
    augmented_cost = torch.cat([upper, lower], dim=-2)
    a = cost.new_full((batch_size, ns + 1), 1.0 / ns)
    b = cost.new_full((batch_size, nv + 1), 1.0 / nv)
    a[:, -1] = dustbin_mass
    b[:, -1] = dustbin_mass
    log_a, log_b = a.log(), b.log()
    log_kernel = log_a.unsqueeze(-1) + log_b.unsqueeze(-2) - augmented_cost / epsilon
    log_u, log_v = torch.zeros_like(a), torch.zeros_like(b)
    sequence_exponent = rho_sequence / (rho_sequence + epsilon)
    visual_exponent = rho_visual / (rho_visual + epsilon)
    for _ in range(n_iters):
        log_u = sequence_exponent * (
            log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        )
        log_v = visual_exponent * (
            log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        )
    return (log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)).exp()


def transport_context(visual_features, augmented_transport, delta=1e-8):
    """Eqs. (4)-(5); aggregate original visual features, not projected ones."""
    transport = augmented_transport[..., :-1, :-1].float()
    real_mass = transport.sum(dim=-1, keepdim=True)
    dustbin_mass = augmented_transport[..., :-1, -1:].float()
    confidence = real_mass / (real_mass + dustbin_mass + delta)
    conditional = transport / (real_mass + delta)
    context = confidence * (conditional @ visual_features.float())
    return context, confidence.squeeze(-1), conditional


def ot_alignment_loss(sequence_projected, visual_projected, augmented_transport,
                      temperature=0.1, delta=1e-8):
    """Eqs. (7)-(9): sum over valid sequence rows, mean over examples."""
    if temperature <= 0:
        raise ValueError("OT alignment temperature must be positive")
    _, confidence, target = transport_context(visual_projected, augmented_transport, delta)
    log_p = F.log_softmax((1.0 - cosine_cost(sequence_projected, visual_projected)) / temperature, dim=-1)
    return -(confidence * (target * log_p).sum(dim=-1)).sum(dim=-1).mean()


def build_labels_from_prompt(input_ids, attention_mask, prompt_attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i in range(labels.shape[0]):
        full_len = int(attention_mask[i].sum().item())
        prompt_len = int(prompt_attention_mask[i].sum().item())
        prompt_end = labels.shape[1] - full_len + prompt_len
        labels[i, :prompt_end] = -100
    return labels


def find_assistant_content_sublist_indexes(token_ids):
    spans = []
    for i in range(len(token_ids) - 1):
        if token_ids[i:i + 2] == [151644, 77091]:
            for j in range(i + 2, len(token_ids)):
                if token_ids[j] == 151645:
                    spans.append((i, j))
                    break
    return spans


def left_pad_tensor_list(tensors, pad_value, dtype=None):
    if not tensors:
        raise ValueError("Expected a non-empty list of tensors")
    padded = torch.full(
        (len(tensors), max(t.shape[0] for t in tensors)), pad_value,
        dtype=dtype or tensors[0].dtype,
    )
    for i, tensor in enumerate(tensors):
        if len(tensor):
            padded[i, -len(tensor):] = tensor
    return padded


def build_text_inputs(texts, processor, is_pc, sequence_prefix_len,
                      visual_memory_prefix_lens=None, sequence_attention_mask=None):
    if visual_memory_prefix_lens is not None and any(visual_memory_prefix_lens):
        raise ValueError("The paper GGM uses fused sequence tokens, without visual-memory tokens")
    tokenized = processor(text=texts, padding=True, return_tensors="pt")
    ids, masks = [], []
    for i in range(len(texts)):
        valid_ids = tokenized["input_ids"][i][tokenized["attention_mask"][i].bool()]
        mask = torch.ones_like(valid_ids)
        if is_pc[i]:
            prefix_ids = valid_ids.new_full((sequence_prefix_len,), processor.tokenizer.pad_token_id)
            prefix_mask = (
                torch.ones_like(prefix_ids) if sequence_attention_mask is None
                else sequence_attention_mask[i].to(mask.dtype)
            )
            valid_ids = torch.cat([prefix_ids, valid_ids])
            mask = torch.cat([prefix_mask, mask])
        ids.append(valid_ids)
        masks.append(mask)
    return {
        "input_ids": left_pad_tensor_list(ids, processor.tokenizer.pad_token_id),
        "attention_mask": left_pad_tensor_list(masks, 0),
    }


def collate(batch, processor, n_points, eval=False, visual_memory_max_spans=0, **kwargs):
    if visual_memory_max_spans:
        raise ValueError("Set visual_memory_max_spans=0 for the paper architecture")
    processor.tokenizer.padding_side = "left"
    is_pc = ["point_cloud" in sample for sample in batch]
    commands = processor(text=[m["sequence"] for m in batch], padding=True, return_tensors="pt")
    messages, prompt_messages = [], []
    for m in batch:
        prompt = [{"role": "user", "content": [{"type": "text", "text": m["description"]}]}]
        prompt_messages.append(prompt)
        messages.append(prompt if eval else prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": m["answer"]}]}
        ])
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=eval) for m in messages]
    packing = dict(
        processor=processor, is_pc=is_pc, sequence_prefix_len=commands["input_ids"].shape[1],
        sequence_attention_mask=commands["attention_mask"],
    )
    inputs = build_text_inputs(texts, **packing)
    inputs.update({
        "sequences": commands["input_ids"],
        "sequence_attention_mask": commands["attention_mask"],
        "point_clouds": torch.stack([
            torch.as_tensor(m["point_cloud"], dtype=torch.float32) if is_pc[i]
            else torch.zeros(n_points, 3) for i, m in enumerate(batch)
        ]),
        "is_pc": torch.tensor(is_pc, dtype=torch.bool),
        "is_img": torch.zeros(len(batch), dtype=torch.bool),
    })
    if eval:
        inputs["sample"] = [m["sample"] for m in batch]
    else:
        prompts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_messages]
        prompt_inputs = build_text_inputs(prompts, **packing)
        inputs["labels"] = build_labels_from_prompt(
            inputs["input_ids"], inputs["attention_mask"], prompt_inputs["attention_mask"],
        )
    return inputs


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=8, include_input=True, include_pi=False):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, points):
        angles = (points[..., None] * self.frequencies).flatten(-2)
        features = [points, angles.sin(), angles.cos()] if self.include_input else [angles.sin(), angles.cos()]
        return torch.cat(features, dim=-1)


def farthest_point_sample_indices(points, num_samples):
    if not 0 < num_samples <= points.shape[1]:
        raise ValueError("Invalid farthest-point sample count")
    batch, count, _ = points.shape
    indices = torch.empty(batch, num_samples, dtype=torch.long, device=points.device)
    distances = torch.full((batch, count), float("inf"), device=points.device)
    farthest = (points.float() - points.float().mean(dim=1, keepdim=True)).square().sum(-1).argmax(-1)
    rows = torch.arange(batch, device=points.device)
    for i in range(num_samples):
        indices[:, i] = farthest
        delta = (points.float() - points[rows, farthest].float().unsqueeze(1)).square().sum(-1)
        distances = torch.minimum(distances, delta)
        farthest = distances.argmax(-1)
    return indices


class FourierPointEncoder(nn.Module):
    """CAD-Recode Fourier-coordinate projection; each sampled point is a region."""
    def __init__(self, hidden_size, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        embedded = self.fourier_embedder(points.float()[..., :3])
        projected = self.projection(embedded.to(self.projection.weight.dtype))
        return projected, points[..., :3]


class VisualFusionHead(nn.Module):
    """Eq. (6): concatenation followed by one linear projection."""
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, sequence_features, visual_context):
        combined = torch.cat([sequence_features, visual_context.to(sequence_features.dtype)], dim=-1)
        return self.projection(combined.to(self.projection.weight.dtype))


class DVACAD(Qwen3VLForConditionalGeneration):
    # Loss is already averaged over examples in this microbatch. Trainer must
    # apply its gradient-accumulation scaling rather than token-count scaling.
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        hidden = config.text_config.hidden_size
        text_embeddings = self.get_text_embed_tokens()
        self.sequence_embed_tokens = nn.Embedding(
            text_embeddings.num_embeddings, hidden, padding_idx=text_embeddings.padding_idx,
        )
        self.point_encoder = FourierPointEncoder(hidden)
        self.sequence_encoder = SequenceEncoder(
            hidden, num_layers=3, num_heads=getattr(config, "sequence_encoder_heads", 8),
            intermediate_size=getattr(config, "sequence_encoder_intermediate_size", 4 * hidden),
        )
        self.sequence_matching_projection = MatchingProjection(hidden)
        self.visual_matching_projection = MatchingProjection(hidden)
        self.visual_fusion_head = VisualFusionHead(hidden)
        self.dustbin_cost = nn.Parameter(torch.tensor(float(getattr(config, "ot_dustbin_cost_init", 1.0))))
        self.ot_contrastive_weight = float(getattr(config, "ot_contrastive_weight", 0.05))
        self.ot_contrastive_temperature = float(getattr(config, "ot_contrastive_temperature", 0.1))
        self.loss_chunk_size = int(os.environ.get("CADRILLE_LOSS_CHUNK_SIZE", "256"))
        self.debug_numerics = os.environ.get("CADRILLE_DEBUG_NUMERICS", "0") == "1"
        self.rope_deltas = None
        self.last_lm_ce_loss = None
        self.last_ot_contrastive_loss = None
        self.last_weighted_ot_contrastive_loss = None
        self.config.ggm_implementation = "tip_dva_202611051433"
        self.config.visual_memory_max_spans = 0
        self.config.heuristic_prior_weight = 0.0
        self.config.ot_contrastive_weight = self.ot_contrastive_weight
        self.config.ot_contrastive_temperature = self.ot_contrastive_temperature
        self.reset_sequence_embed_tokens_from_text()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        want_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        if "sequence_embed_tokens.weight" in loading_info["missing_keys"]:
            model.reset_sequence_embed_tokens_from_text()
        return (model, loading_info) if want_loading_info else model

    def get_text_embed_tokens(self):
        language_model = getattr(self.model, "language_model", None)
        return language_model.embed_tokens if language_model is not None else self.get_input_embeddings()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif module is self and hasattr(self, "dustbin_cost"):
            with torch.no_grad():
                self.dustbin_cost.fill_(float(getattr(self.config, "ot_dustbin_cost_init", 1.0)))

    def reset_sequence_embed_tokens_from_text(self):
        with torch.no_grad():
            self.sequence_embed_tokens.weight.copy_(self.get_text_embed_tokens().weight)

    def freeze_pc(self):
        self.point_encoder.requires_grad_(False)

    def assert_finite(self, name, tensor):
        if self.debug_numerics and tensor is not None and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite tensor detected at {name}, shape={tuple(tensor.shape)}")

    def construct_cost_matrix(self, token_embeds, patch_embeds, lambda_spatial=0.0, **kwargs):
        if lambda_spatial:
            raise ValueError("Paper Eq. (1) uses only projected cosine distance")
        sequence = self.sequence_matching_projection(token_embeds)
        visual = self.visual_matching_projection(patch_embeds.to(token_embeds.dtype))
        return cosine_cost(sequence, visual)

    def sinkhorn_transport_matrix(self, cost_matrix, epsilon=0.1, n_iters=80, return_augmented=False, **kwargs):
        with torch.autocast(device_type=cost_matrix.device.type, enabled=False):
            if self.training and torch.is_grad_enabled():
                plan = checkpoint(
                    unbalanced_sinkhorn, cost_matrix, self.dustbin_cost,
                    epsilon=epsilon, n_iters=n_iters, use_reentrant=False,
                )
            else:
                plan = unbalanced_sinkhorn(cost_matrix, self.dustbin_cost, epsilon=epsilon, n_iters=n_iters)
        return plan if return_augmented else plan[..., :-1, :-1]

    def ground_sequence(self, sequence_features, visual_features, compute_alignment=True):
        seq_projected = self.sequence_matching_projection(sequence_features)
        vis_projected = self.visual_matching_projection(visual_features.to(sequence_features.dtype))
        with torch.autocast(device_type=sequence_features.device.type, enabled=False):
            cost = cosine_cost(seq_projected, vis_projected)
            plan = self.sinkhorn_transport_matrix(cost, return_augmented=True)
            visual_context, confidence, _ = transport_context(visual_features, plan)
            alignment_loss = ot_alignment_loss(
                seq_projected, vis_projected, plan, self.ot_contrastive_temperature,
            ) if compute_alignment else None
        fused = self.visual_fusion_head(sequence_features, visual_context)
        return fused, alignment_loss, plan, confidence

    def _sequence_cross_entropy(self, hidden_states, labels):
        # Eqs. (19)-(20) sum target-token NLL within each example, then average
        # examples. Checkpointing each vocabulary projection bounds peak memory.
        hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
        targets = labels[:, 1:].reshape(-1).to(hidden.device)
        valid = targets != -100
        hidden, targets = hidden[valid], targets[valid]
        chunk_size = self.loss_chunk_size if self.loss_chunk_size > 0 else max(len(targets), 1)

        def chunk_loss(features, token_ids):
            logits = self.lm_head(features)
            self.assert_finite("lm_head.logits_chunk", logits)
            return F.cross_entropy(logits.float(), token_ids, reduction="sum")

        total = hidden_states.sum() * 0.0
        for start in range(0, len(targets), chunk_size):
            features, token_ids = hidden[start:start + chunk_size], targets[start:start + chunk_size]
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, features, token_ids, use_reentrant=False)
            else:
                total = total + chunk_loss(features, token_ids)
        return total / hidden_states.shape[0]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, rope_deltas=None,
                cache_position=None, logits_to_keep=0, point_clouds=None, sequences=None,
                sequence_attention_mask=None, is_pc=None, is_img=None,
                return_ot_contrastive_loss=False, **kwargs):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        self.last_lm_ce_loss = self.last_ot_contrastive_loss = self.last_weighted_ot_contrastive_loss = None
        alignment_losses = []
        compute_alignment = (labels is not None and self.ot_contrastive_weight > 0) or return_ot_contrastive_loss
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            prefill = past_key_values is None or past_key_values.get_seq_length() == 0
            if cache_position is not None and cache_position.numel():
                prefill = int(cache_position[0]) == 0
            if is_pc is not None and torch.any(is_pc) and prefill:
                if sequences is None or point_clouds is None:
                    raise ValueError("GGM requires both the original CAD sequence and its point cloud")
                sequence_mask = torch.ones_like(sequences, dtype=torch.bool) if sequence_attention_mask is None else sequence_attention_mask.bool()
                sequence_features = self.sequence_encoder(self.sequence_embed_tokens(sequences), sequence_mask)
                visual_features, _ = self.point_encoder(point_clouds)
                self.assert_finite("sequence_encoder", sequence_features)
                self.assert_finite("point_encoder", visual_features)
                for i in range(len(sequences)):
                    if not is_pc[i]:
                        continue
                    features = sequence_features[i, sequence_mask[i]].unsqueeze(0)
                    fused, alignment, _, _ = self.ground_sequence(features, visual_features[i:i + 1], compute_alignment)
                    self.assert_finite("geometry_aware_sequence", fused)
                    if alignment is not None:
                        alignment_losses.append(alignment)
                    start = int(attention_mask[i].nonzero()[0]) if attention_mask is not None else 0
                    end = start + fused.shape[1]
                    if end > inputs_embeds.shape[1]:
                        raise ValueError("Prompt has insufficient prefix slots for the fused source sequence")
                    inputs_embeds[i, start:end] = fused[0].to(inputs_embeds.dtype)
        alignment_loss = torch.stack(alignment_losses).mean() if alignment_losses else inputs_embeds.sum() * 0
        if compute_alignment:
            self.last_ot_contrastive_loss = alignment_loss.detach()
            self.last_weighted_ot_contrastive_loss = (self.ot_contrastive_weight * alignment_loss).detach()
        if return_ot_contrastive_loss and labels is None:
            if not return_dict:
                return (alignment_loss,)
            return Qwen3VLCausalLMOutputWithPast(loss=alignment_loss)
        if rope_deltas is not None:
            self.model.rope_deltas = rope_deltas
        kwargs.pop("num_items_in_batch", None)
        outputs = self.model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=False if labels is not None else use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, cache_position=cache_position, **kwargs,
        )
        hidden_states = outputs[0]
        self.assert_finite("qwen.hidden_states", hidden_states)
        loss, logits = None, None
        if labels is not None:
            loss = self._sequence_cross_entropy(hidden_states, labels)
            self.last_lm_ce_loss = loss.detach()
            loss = loss + self.ot_contrastive_weight * alignment_loss
        else:
            selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
            logits = self.lm_head(hidden_states[:, selection])
            self.assert_finite("lm_head.logits", logits)
        if not return_dict:
            result = (logits,) + outputs.to_tuple()[1:]
            return (loss,) + result if loss is not None else result
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        for name in ("point_clouds", "sequences", "sequence_attention_mask", "is_pc", "is_img"):
            inputs[name] = kwargs.get(name)
        return inputs
