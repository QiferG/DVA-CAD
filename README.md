# DVA-CAD

This project builds on [cadrille-rl](https://github.com/marinabar/cadrille-rl).
The upstream license and attribution are preserved in `LICENSE` and `NOTICE`.

## SFT Training

Four-GPU training, keeping the effective batch size at 32:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CADRILLE_LOSS_CHUNK_SIZE=64 \
torchrun --standalone --nproc-per-node=4 seqeditor_train.py \
  --model-path "$MODEL_DIR" \
  --data-path "$DATA_DIR" \
  --log-path "$SFT_DIR" \
  --representation command \
  --global-batch-size 32 \
  --per-device-batch-size 1 \
  --max-steps 60000 \
  --learning-rate 1e-4
```
## Test Generation

```bash
CUDA_VISIBLE_DEVICES=0 python seqeditor_test.py \
  --model-path "$SFT_DIR/seqeditor_model" \
  --data-path "$DATA_DIR" \
  --out-path "$TEST_DIR/test.jsonl" \
  --num_samples 1 \
  --batch_size 1 \
  --max_new_tokens 2048
```
## GRPO Reinforcement Learning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 rl_finetune/train_cadrille_grpo.py \
  --sft_path "$SFT_DIR/seqeditor_model" \
  --data_path "$DATA_DIR" \
  --save_path "$RL_DIR" \
  --representation command \
  --batch_size 4 \
  --num_generations 6 \
  --top_samples 6 \
  --learning_rate 1e-5 \
  --max_optimizer_steps 2000 \
  --grid_resolution 64 \
  --tau_phi 0.01 \
  --lambda_edit 1.5 \
  --lambda_stable 1.0 \
  --save_every_steps 0 \
  --save_every_optimizer_steps 100
```
