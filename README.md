# Logger: Simple Solar Autoencoder + WandB Logging

This repo contains a one-file training example (`logger.py`) that trains a small convolutional autoencoder on solar images and logs reconstructions with SunPy panels. Optional experiment tracking is provided via Weights & Biases (WandB).

## Requirements
- Python 3.10+
- pip

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: `sunpy[all]` may take a while and may require extra system packages depending on your OS.

## Data format
Provide two NumPy files:
- `images.npy`: shape `(N, H, W)` or `(N, H, W, 1)` with values roughly in [0..255]
- `headers.npy`: shape `(N,)` array of dicts (SunPy-compatible map headers). If present, `T_OBS` is shown in panel titles.

## WandB API key (.env)
Only the WandB API key should be stored in an `.env` file. If WandB logging is enabled and no key is found, the script will securely prompt for it.

Create `.env` at the project root (choose one of the following):

- Windows PowerShell:
```powershell
Set-Content -Path .env -Value "WANDB_API_KEY=your_wandb_key_here"
```

- Bash (Linux/macOS/Git Bash):
```bash
echo "WANDB_API_KEY=your_wandb_key_here" > .env
```

`.gitignore` is set to exclude `.env` from commits.

## Quick start
Basic run (auto-selects CUDA if available, enables WandB by default):
```bash
python logger.py
```
You will be prompted for the WandB API key if not set in `.env`.

## CLI usage
All runtime options are available as CLI flags. Defaults are shown in parentheses.

- Device and performance:
  - `--device {auto|cuda|cpu}` (auto)
  - `--pin-memory` / `--no-pin-memory` (default auto: true when device is cuda)
  - `--num-workers INT` (2)

- Data:
  - `--images-path PATH` (/content/drive/MyDrive/wind_data/images.npy)
  - `--headers-path PATH` (/content/drive/MyDrive/wind_data/headers.npy)

- Training:
  - `--batch-size INT` (8)
  - `--lr FLOAT` (1e-3)
  - `--epochs INT` (3)

- WandB:
  - `--wandb` / `--no-wandb` (enabled by default)
  - `--wandb-project NAME` (solar-ae-demo)
  - `--wandb-run-name NAME` (defaults to `simple-ae-<device>`)

Examples:
```bash
# Use defaults; will prompt for API key if not in .env
python logger.py

# Explicit dataset paths and training options
python logger.py \
  --device cuda \
  --images-path "C:\\data\\images.npy" \
  --headers-path "C:\\data\\headers.npy" \
  --batch-size 16 --lr 5e-4 --epochs 10 --num-workers 4 --pin-memory \
  --wandb --wandb-project myproj --wandb-run-name exp1

# Disable WandB entirely
python logger.py --no-wandb
```

## Outputs
- Checkpoint: `checkpoints/autoencoder_final.pt`
- If WandB is enabled: metrics, reconstruction panels, and checkpoint artifact are logged to your project.

## Troubleshooting
- "CUDA not available": the script will fall back to CPU (use `--device cpu`).
- WandB login failed: ensure your key is correct in `.env` or at the prompt; you can also disable logging with `--no-wandb`.
- SunPy/image plotting issues: verify your headers are valid SunPy map headers and image arrays are in the expected shape/value range.

---
Happy training!
