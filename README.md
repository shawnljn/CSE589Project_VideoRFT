# VideoRFT (class project)

This repository packages the VideoRFT codebase plus our class additions for evaluating multimodal reasoning models on video benchmarks. The focus of this handoff is to make the repo easy to review on GitHub while keeping large assets out of version control.

## Repo highlights
- Core training/eval code from the VideoRFT release under `src/r1-v` and `src/qwen-vl-utils`.
- Evaluation metadata under `Evaluation/` (video files are not tracked).
- New utilities added for the class project:
  - `src/eval_bench.py` (updated evaluation loop and reward handling).
  - `src/run_ablation.py` for quick decoding sweeps.
  - `src/compare_results.py`, `src/visualize_results.py`, and `src/export_examples.py` for summarizing and visualizing evaluation outputs.
  - `download_vids.py` to pull evaluation videos listed in `Evaluation/eval_videommmu.json`.
  - Sample outputs under `eval_results/` for reference.

## Prerequisites
- Python 3.11
- CUDA-capable GPU (tested with CUDA 12.8). Set `TORCH_DEVICE=cpu` in the setup script if you need CPU-only installs.
- FFmpeg is recommended for video handling; `yt-dlp` plus browser cookies are needed only if you want to download evaluation videos yourself.

## Environment setup
From the repo root:

```bash
./setup_env.sh                     # builds .venv and installs pinned deps
# optional tweaks
# TORCH_DEVICE=cpu ./setup_env.sh  # CPU-only install
# INSTALL_FLASH_ATTN=0 ./setup_env.sh  # skip flash-attn build
source .venv/bin/activate
```

Key packages are pinned: `torch==2.5.1`, `transformers==4.51.3`, `trl==0.16.0`, `vllm==0.7.3`, `wandb==0.18.3`, plus editable installs of `src/r1-v` and `src/qwen-vl-utils` (with decord extras).

## Data and large artifacts
- `.gitignore` excludes `checkpoints/`, `Evaluation/` video data, `vllm/`, media files (`*.mp4`, etc.), and other heavyweight artifacts. Keep these paths untracked before pushing to GitHub.
- Place any model weights you want to test under `checkpoints/` (or point `--model_path` to a Hugging Face model ID).
- To fetch evaluation videos referenced in `Evaluation/eval_videommmu*.json`, run:
  ```bash
  python download_vids.py
  ```
  Configure cookies via `YT_COOKIES` if the videos require authentication.

# Download weights into checkpoints/

Install HF tooling and (optionally) login:

```pip install -U huggingface_hub
huggingface-cli login   # optional; required only for gated models (not expected here)
```

Download the RL-tuned model (VideoRFT-3B):

```huggingface-cli download QiWang98/VideoRFT-3B \
  --local-dir checkpoints/VideoRFT-3B \
  --local-dir-use-symlinks False
```

(Alternative) Download the SFT-only checkpoint if you want to compare:

```huggingface-cli download QiWang98/VideoRFT-SFT-3B \
  --local-dir checkpoints/VideoRFT-SFT-3B \
  --local-dir-use-symlinks False
```

Then run evaluation with local weights:

```python src/eval_bench.py \
  --model_path checkpoints/VideoRFT-3B \
  --file_name videommmu_subset
```

## Running evaluation and analyses
- Single evaluation run (defaults to `Evaluation/eval_videommmu_subset.json` inside the script):
  ```bash
  python src/eval_bench.py --model_path checkpoints/VideoRFT-3B --file_name videommmu_subset
  ```
  Environment overrides:
  - `TP_SIZE` (tensor parallelism, default 1)
  - `GPU_MEM_UTIL` (vLLM memory fraction, default 0.8)
  - `TEMP`, `TOP_P`, `MAX_TOKENS` (decoding parameters)

- Decoding ablations:
  ```bash
  python src/run_ablation.py --model_path checkpoints/VideoRFT-3B --file_name videommmu_subset
  ```
  Edit `CONFIGS` inside the script to try new temperature/top-p combinations.

- Summaries and plots from saved JSON outputs:
  ```bash
  python src/compare_results.py --inputs eval_results/eval_videommmu_videommmu_subset_greedy_output.json --out eval_results/report.txt
  python src/visualize_results.py --input eval_results/eval_videommmu_videommmu_subset_greedy_output.json
  ```

- Export qualitative examples (frames plus markdown):
  ```bash
  python src/export_examples.py --input eval_results/eval_videommmu_videommmu_subset_greedy_output.json --output-dir eval_results/examples --num 6
  ```

## References
- Paper: https://arxiv.org/abs/2505.12434
- Datasets: https://huggingface.co/datasets/QiWang98/VideoRFT-Data
- Models: https://huggingface.co/QiWang98/VideoRFT
