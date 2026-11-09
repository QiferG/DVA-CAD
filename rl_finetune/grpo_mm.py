import os

import wandb
import time

os.environ["PYGLET_HEADLESS"] = "True"

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.
    """
    # Avoid materializing the full `log_softmax(logits)` tensor, which doubles
    # the activation memory over the vocabulary dimension. We only need the
    # target-token log-probabilities:
    #   log p(y) = logits[y] - logsumexp(logits)
    selected_logits = logits.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1).float()

    vocab_chunk_size = 8192
    normalization = None
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        chunk_logsumexp = torch.logsumexp(logits[..., start:end].float(), dim=-1)
        normalization = chunk_logsumexp if normalization is None else torch.logaddexp(normalization, chunk_logsumexp)

    log_probs = selected_logits - normalization
    if not torch.isfinite(log_probs).all():
        raise RuntimeError("Non-finite policy log probabilities; check model weights and gradients")
    return log_probs


def compute_log_probs(model, batch, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.
    """
    input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw, sequences, sequence_attention_mask = batch
    # We only need next-token logits for the completion suffix. Because the model
    # slices hidden states before `lm_head`, ask it for just the last
    # `logits_to_keep + 1` positions so that, after dropping the final position
    # for next-token alignment, we still retain exactly `logits_to_keep` logits.
    logits_to_return = logits_to_keep + 1 if isinstance(logits_to_keep, int) and logits_to_keep > 0 else 0
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        point_clouds=point_cloud,
        is_pc=is_pc,
        is_img=is_img,
        sequences=sequences,
        sequence_attention_mask=sequence_attention_mask,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        logits_to_keep=logits_to_return,
        use_cache=False,
    ).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


def compute_log_probs_chunked(model, batch, logits_to_keep, chunk_size):
    """
    Computes log probabilities in smaller batch chunks to reduce peak memory.
    """
    if chunk_size is None or chunk_size <= 0:
        return compute_log_probs(model, batch, logits_to_keep)

    batch_size = batch[0].shape[0]
    if batch_size <= chunk_size:
        return compute_log_probs(model, batch, logits_to_keep)

    outputs = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        batch_chunk = tuple(
            None if tensor is None else tensor[start:end]
            for tensor in batch
        )
        outputs.append(compute_log_probs(model, batch_chunk, logits_to_keep))
    return torch.cat(outputs, dim=0)


def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def generate_completions(
    model,
    processor,
    inputs,
    num_generations=4,
    max_completion_length=32,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
    top_k=50,
):
    """
    Generates multiple completions for each prompt.
    """
    device = model.device
    prompt_ids = inputs["input_ids"].to(device, non_blocking=True)
    prompt_mask = inputs["attention_mask"].to(device, non_blocking=True)
    point_cloud = inputs["point_clouds"].to(device, non_blocking=True)
    is_pc = inputs["is_pc"].to(device, non_blocking=True)
    is_img = inputs["is_img"].to(device, non_blocking=True)
    sequences = inputs["sequences"].to(device, non_blocking=True)
    sequences_attention_mask = inputs["sequence_attention_mask"].to(device, non_blocking=True)
    pixel_values_videos = inputs['pixel_values_videos'].to(device, non_blocking=True) if inputs.get('pixel_values_videos',
                                                                                               None) is not None else None
    video_grid_thw = inputs['video_grid_thw'].to(device, non_blocking=True) if inputs.get('video_grid_thw',
                                                                                     None) is not None else None
    prompt_length = prompt_ids.size(1)
    batch_size = prompt_ids.size(0)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    point_cloud = point_cloud.repeat_interleave(num_generations, dim=0)
    is_pc = is_pc.repeat_interleave(num_generations, dim=0)
    is_img = is_img.repeat_interleave(num_generations, dim=0)
    sequences = sequences.repeat_interleave(num_generations, dim=0)
    sequences_attention_mask = sequences_attention_mask.repeat_interleave(num_generations, dim=0)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.repeat_interleave(num_generations, dim=0)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.repeat_interleave(num_generations, dim=0)

    if sequences is None:
        print("sequences is None")

    print(
        "Generating "
        f"{num_generations} completions for each of {batch_size} prompts. "
        f"model.training={model.training}, "
        f"prompt_shape={tuple(prompt_ids.shape)}, "
        f"sequence_shape={tuple(sequences.shape) if sequences is not None else None}, "
        f"point_cloud_shape={tuple(point_cloud.shape)}, "
        f"is_pc_sum={int(is_pc.sum().item())}, "
        f"is_img_sum={int(is_img.sum().item())}",
        flush=True,
    )
    safe_temperature = max(float(temperature), 1e-5)
    outputs = model.generate(input_ids=prompt_ids,
                             attention_mask=prompt_mask,
                             point_clouds=point_cloud,
                             is_pc=is_pc,
                             is_img=is_img,
                             sequences=sequences,
                             sequence_attention_mask=sequences_attention_mask,
                             pixel_values_videos=pixel_values_videos,
                             video_grid_thw=video_grid_thw,
                             max_new_tokens=max_completion_length,
                             use_cache=True,
                             do_sample=do_sample,
                             temperature=safe_temperature,
                             top_p=top_p,
                             top_k=top_k,
                             renormalize_logits=True,
                             remove_invalid_values=True,
                             early_stopping=False,
                             bad_words_ids=[
                                        [model.config.image_token_id],
                                        [model.config.video_token_id],
                                        [model.config.vision_start_token_id],
                                        [model.config.vision_end_token_id],]
                             )
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, processor.tokenizer.eos_token_id)
    return point_cloud, prompt_ids, prompt_mask, is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask, sequences, sequences_attention_mask


def generate_rollout_data(model, reward_function,
                          processor, batch_samples, num_generations, max_completion_length, top_samples=None,
                          temperature=1.0, top_p=1.0, do_sample=True, top_k=50):
    """
    Generates data for GRPO rollouts including completions and log probabilities.
    """
    prompts = batch_samples
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            point_cloud, prompt_ids, prompt_mask, is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask, sequences, sequences_attention_mask = generate_completions(
                model,
                processor,
                prompts,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            top_k=top_k,
        )

        gen_time = time.perf_counter() - t0
        print(f"[TIME] generation time: {gen_time:.3f} s", flush=True)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        formatted_completions = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        repeated_answers = [a['edited_pic_name'] for a in batch_samples['sample'] for _ in range(num_generations)]
        originals = [o['original_pic_name'] for o in batch_samples['sample'] for _ in range(num_generations)]

        t1 = time.perf_counter()

        rewards = torch.tensor(
            reward_function(completions=formatted_completions, answer=repeated_answers, originals=originals),
            dtype=torch.float32,
            device=model.device
        )
        reward_time = time.perf_counter() - t1
        print(f"[TIME] reward computation time: {reward_time:.3f} s", flush=True)
        print("Rewards", rewards, flush=True)

        batch_size = len(prompts['input_ids'])
        if top_samples is None:
            top_samples = num_generations
        rewards = rewards.view(batch_size, num_generations)
        reward_std = rewards.std(dim=1, keepdim=True, unbiased=False)
        has_reward_signal = reward_std > 1e-8
        avg_reward = rewards.mean().item()
        print("Average Reward:", avg_reward, flush=True)
        reward_mean = rewards.mean(dim=1, keepdim=True)
        reward_std = reward_std.clamp_min(1e-4)
        advantages_all = (rewards - reward_mean) / reward_std
        advantages_all = torch.where(has_reward_signal, advantages_all, torch.zeros_like(advantages_all))
        print(
            f"Reward std mean: {reward_std.mean().item():.6f}, "
            f"Advantage std: {advantages_all.std(unbiased=False).item():.6f}",
            flush=True,
        )

        selection_top_k = min(top_samples, num_generations)
        per_prompt_top_samples = torch.full(
            (batch_size,),
            selection_top_k,
            dtype=torch.long,
            device=rewards.device,
        )
        max_selected_samples = selection_top_k
        _, sorted_indices = torch.topk(advantages_all, selection_top_k, dim=1)

        selected_flattened_indices = []
        selected_advantages = []
        for row_idx in range(batch_size):
            num_selected = int(per_prompt_top_samples[row_idx].item())
            chosen_indices = sorted_indices[row_idx, :num_selected]
            flat_indices = row_idx * num_generations + chosen_indices
            selected_flattened_indices.append(flat_indices)
            chosen_advantages = advantages_all[row_idx, chosen_indices]
            selected_advantages.append(chosen_advantages)

        flattened_indices = torch.cat(selected_flattened_indices, dim=0)
        advantages = torch.cat(selected_advantages, dim=0).unsqueeze(1)

        selected_count = flattened_indices.shape[0]
        input_ids = input_ids[flattened_indices].reshape(selected_count, *input_ids.shape[1:])
        attention_mask = attention_mask[flattened_indices].reshape(selected_count, *attention_mask.shape[1:])
        point_cloud = point_cloud[flattened_indices].reshape(selected_count, *point_cloud.shape[1:])
        completion_mask = completion_mask[flattened_indices].reshape(selected_count, *completion_mask.shape[1:])
        is_pc = is_pc[flattened_indices].reshape(selected_count, *is_pc.shape[1:])
        is_img = is_img[flattened_indices].reshape(selected_count, *is_img.shape[1:])
        sequences = sequences[flattened_indices].reshape(selected_count, *sequences.shape[1:])
        sequences_attention_mask = sequences_attention_mask[flattened_indices].reshape(selected_count, *sequences_attention_mask.shape[1:])
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos[flattened_indices].reshape(selected_count, *pixel_values_videos.shape[1:])
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw[flattened_indices].reshape(selected_count, *video_grid_thw.shape[1:])
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "formatted_completions": formatted_completions,
            "repeated_answers": repeated_answers,
            "logits_to_keep": logits_to_keep,
            "batch_size": len(prompts['input_ids']),
            "num_generations": num_generations,
            "selected_samples_per_prompt": per_prompt_top_samples.tolist(),
            "max_selected_samples_per_prompt": max_selected_samples,
            "point_cloud": point_cloud,
            "sequences": sequences,
            "sequence_attention_mask": sequences_attention_mask,
            "advantages": advantages,
            "is_pc": is_pc,
            "is_img": is_img,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        with torch.no_grad():
            old_log_probs = compute_log_probs_chunked(
                model,
                (input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw, sequences, sequences_attention_mask),
                logits_to_keep,
                chunk_size=1,
            )
        result["old_log_probs"] = old_log_probs.detach()
    finally:
        if was_training:
            model.train()
    return result, avg_reward


def grpo_loss(
    model,
    rollout_data,
    ref_model=None,
    kl_coef=0.0,
    epsilon_high=0.2,
    epsilon_low=0.2,
    log_context=None,
):
    """
    Computes the GRPO loss for updating the policy model.
    """
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    sequences = rollout_data["sequences"]
    sequence_attention_mask = rollout_data["sequence_attention_mask"]
    token_log_probs = compute_log_probs_chunked(
        model,
        (input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw, sequences, sequence_attention_mask),
        logits_to_keep,
        chunk_size=1,
    )
    log_ratio = token_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Non-finite GRPO policy ratio")
    valid_completion_mask = completion_mask.bool()
    completion_lengths = completion_mask.sum(dim=1).clamp_min(1)
    masked_ratio_mean = (
        (ratio * completion_mask).sum(dim=1) / completion_lengths
    ).mean().item()
    valid_log_ratio = log_ratio[valid_completion_mask]
    if valid_log_ratio.numel() > 0:
        log_ratio_mean = valid_log_ratio.mean().item()
        log_ratio_std = valid_log_ratio.std(unbiased=False).item()
        log_ratio_min = valid_log_ratio.min().item()
        log_ratio_max = valid_log_ratio.max().item()
    else:
        log_ratio_mean = 0.0
        log_ratio_std = 0.0
        log_ratio_min = 0.0
        log_ratio_max = 0.0
    kl_term = None
    if ref_model is not None and kl_coef > 0:
        with torch.no_grad():
            ref_log_probs = compute_log_probs_chunked(
                ref_model,
                (input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw, sequences, sequence_attention_mask),
                logits_to_keep,
                chunk_size=1,
            )
        log_reference_ratio = ref_log_probs - token_log_probs
        kl_term = torch.exp(log_reference_ratio) - log_reference_ratio - 1.0
    if log_context is None:
        print(
            f"ratio mean {ratio.mean()}, "
            f"masked ratio mean {masked_ratio_mean:.6f}, "
            f"log_ratio mean {log_ratio_mean:.6f}, "
            f"log_ratio std {log_ratio_std:.6f}, "
            f"log_ratio min {log_ratio_min:.6f}, "
            f"log_ratio max {log_ratio_max:.6f}"
        )
    else:
        kl_mean = (
            ((kl_term * completion_mask).sum(dim=1) / completion_lengths).mean().item()
            if kl_term is not None else 0.0
        )
        print(
            f"Epoch {log_context['epoch']}/{log_context['train_epochs']}, "
            f"Step {log_context['step']}/{log_context['total_steps']}, "
            f"Update {log_context['update']}/{log_context['updates_per_step']}, "
            f"ratio mean {ratio.mean()}, "
            f"masked ratio mean {masked_ratio_mean:.6f}, "
            f"log_ratio mean {log_ratio_mean:.6f}, "
            f"log_ratio std {log_ratio_std:.6f}, "
            f"log_ratio min {log_ratio_min:.6f}, "
            f"log_ratio max {log_ratio_max:.6f}, "
            f"kl mean {kl_mean:.6f}"
        )
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    per_token_loss = surrogate_loss
    seq_rl_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_lengths
    loss = -seq_rl_loss.mean()
    if kl_term is not None:
        seq_kl = (kl_term * completion_mask).sum(dim=1) / completion_lengths
        loss = loss + kl_coef * seq_kl.mean()
    return loss


def save_checkpoint(model, processor, optimizer, save_dir, step, epoch, tag):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, tag)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    return ckpt_dir


def save_checkpoint_with_log(
    model,
    processor,
    optimizer,
    save_dir,
    checkpoint_prefix,
    step,
    epoch,
    tag_suffix,
    label,
):
    ckpt_dir = save_checkpoint(
        model,
        processor,
        optimizer,
        save_dir,
        step=step,
        epoch=epoch,
        tag=f"{checkpoint_prefix}_{tag_suffix}",
    )
    print(f"[checkpoint] saved {label} checkpoint to {ckpt_dir}", flush=True)
    return ckpt_dir


def train_with_grpo_mm(model, processor, train_data, sampler, batch_size=4,
                    num_generations=4, top_samples=None, max_completion_length=128,
                    learning_rate=5e-6, batch_updates=3, epsilon_high=0.2, epsilon_low=0.2, train_epochs=1,
                    reward_function=None, collate_fn=None, run_id=None,
                    ref_model=None, kl_coef=0.0,
                    save_path="./models", temperature=1.0, top_p=1.0, do_sample=True, top_k=50,
                    dataloader_workers=0, pin_memory=True,
                    persistent_workers=True, save_every_steps=0, save_mid_epoch=True,
                    save_every_optimizer_steps=0, epoch_save=1, save_prefix=None,
                    gradient_accumulation_steps=1, max_optimizer_steps=2000):

    rank = dist.get_rank()

    if top_samples is None:
        top_samples = num_generations
    if top_samples != num_generations:
        raise ValueError("Paper RL uses GRPO on all candidates; top_samples must equal num_generations")

    step = 0
    optimizer_step = 0
    accumulation_counter = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint_prefix = run_id or save_prefix or "checkpoint"

    model.train()
    for epoch in range(train_epochs):
        sampler.set_epoch(epoch)
        dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and dataloader_workers > 0,
        )
        print(f"\nEpoch {epoch + 1}/{train_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_samples in enumerate(dataloader, start=1):
            current_step = step + 1
            rollout_data, avg_reward = generate_rollout_data(
                model.module,
                reward_function,
                processor,
                batch_samples,
                num_generations,
                max_completion_length,
                top_samples=top_samples,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                top_k=top_k,
            )
            for grpo_iter in range(batch_updates):
                log_context = {
                    "epoch": epoch + 1,
                    "train_epochs": train_epochs,
                    "step": current_step,
                    "total_steps": len(dataloader),
                    "update": grpo_iter + 1,
                    "updates_per_step": batch_updates,
                }
                # Complete each DDP forward/backward before the next candidate.
                # This also bounds activation memory while retaining all six.
                selected_count = rollout_data["input_ids"].shape[0]
                loss = torch.zeros((), device=next(model.parameters()).device)
                for sample_index in range(selected_count):
                    micro_rollout = {
                        key: value[sample_index:sample_index + 1]
                        if torch.is_tensor(value) and value.ndim and value.shape[0] == selected_count
                        else value for key, value in rollout_data.items()
                    }
                    micro_loss = grpo_loss(
                        model, micro_rollout,
                        ref_model=ref_model, kl_coef=kl_coef,
                        epsilon_high=epsilon_high, epsilon_low=epsilon_low,
                        log_context=log_context,
                    )
                    (micro_loss / (selected_count * max(gradient_accumulation_steps, 1))).backward()
                    loss += micro_loss.detach() / selected_count
                accumulation_counter += 1
                is_last_backward = (batch_index == len(dataloader)) and (grpo_iter + 1 == batch_updates)
                should_step = (
                    accumulation_counter % max(gradient_accumulation_steps, 1) == 0
                ) or is_last_backward
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                # Log to wandb
                if rank == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "step": current_step,
                        "optimizer_step": optimizer_step,
                        "update": grpo_iter + 1,
                        "epoch": epoch + 1,
                    })
                    print(f"Epoch {epoch + 1}/{train_epochs}, Step {current_step}/{len(dataloader)}, "
                          f"Update {grpo_iter + 1}/{batch_updates}, loss: {loss.item():.4f}, "
                          f"optimizer_step: {optimizer_step}")
            step = current_step

            if (
                rank == 0
                and save_mid_epoch
                and save_every_steps > 0
                and optimizer_step > 0
                and optimizer_step % save_every_steps == 0
                and should_step
            ):
                save_checkpoint_with_log(
                    model.module,
                    processor,
                    optimizer,
                    save_path,
                    checkpoint_prefix,
                    step=step,
                    epoch=epoch + 1,
                    tag_suffix=f"epoch{epoch + 1}_optstep{optimizer_step}",
                    label="step",
                )

            if (
                rank == 0
                and save_every_optimizer_steps > 0
                and optimizer_step > 0
                and optimizer_step % save_every_optimizer_steps == 0
                and should_step
            ):
                save_checkpoint_with_log(
                    model.module,
                    processor,
                    optimizer,
                    save_path,
                    checkpoint_prefix,
                    step=step,
                    epoch=epoch + 1,
                    tag_suffix=f"epoch{epoch + 1}_optstep{optimizer_step}",
                    label="optimizer-step",
                )

            if max_optimizer_steps > 0 and optimizer_step >= max_optimizer_steps:
                break

        if rank == 0:
            if epoch_save > 0 and (epoch + 1) % epoch_save == 0:
                save_checkpoint_with_log(
                    model.module,
                    processor,
                    optimizer,
                    save_path,
                    checkpoint_prefix,
                    step=step,
                    epoch=epoch + 1,
                    tag_suffix=f"epoch{epoch + 1}_final",
                    label="epoch",
                )

        dist.barrier()
        if max_optimizer_steps > 0 and optimizer_step >= max_optimizer_steps:
            if rank == 0:
                save_checkpoint_with_log(
                    model.module, processor, optimizer, save_path, checkpoint_prefix,
                    step=step, epoch=epoch + 1,
                    tag_suffix=f"final_optstep{optimizer_step}", label="final",
                )
            dist.barrier()
            break
    return model.module
