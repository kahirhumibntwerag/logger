

import io
import os
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import sunpy.map

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dotenv import load_dotenv

# Optional: Weights & Biases logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# Load environment variables from a local .env file if present
load_dotenv()

# -----------------------
# Constants and utilities
# -----------------------
LOG_EPS = 1e-5


# -----------------------
# Time/title helpers
# -----------------------
def _format_t_obs(t_obs):
    if t_obs is None:
        return None
    s = str(t_obs).strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return s
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# -----------------------
# Transforms
# -----------------------
def to_float32(image: np.ndarray):
    return image.astype(np.float32)

def normalize_255_to_m11(image: np.ndarray):
    # Expect value range ~ [0,255], map to [-1,1]
    return (image / 255.0) * 2.0 - 1.0

def denormalize_m11_to_255(t: torch.Tensor):
    # Input tensor in [-1,1], output in [0,255]
    return (t + 1.0) / 2.0 * 255.0

def denormalize_from_0_255_to_log_space(x: torch.Tensor | np.ndarray):
    """
    x: tensor/ndarray in [0,255]
    Returns numpy array in log space.
    """
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().to('cpu').numpy()
    else:
        x_cpu = np.asarray(x)
    return np.log10(LOG_EPS) + (x_cpu / 255.0) * (np.log10(20000) - np.log10(LOG_EPS))

def inverse_log_transform(x: np.ndarray):
    # x: numpy array in log space
    restored = (10.0 ** x) - LOG_EPS
    restored = np.clip(restored, 0.0, np.inf)
    return restored.astype(np.float32)

# Public transform pipeline for dataset (HxW -> 1xHxW, float in [-1,1])
solar_transform = transforms.Compose([
    to_float32,
    normalize_255_to_m11,  # [-1,1] float
    transforms.ToTensor(), # HxW -> 1xHxW
])


# -----------------------
# Dataset + Collate
# -----------------------
class SolarWindDataset(Dataset):
    """
    images.npy: array-like of HxW (or HxWx1) images, values ~ [0..255]
    headers.npy: array-like of dict headers (SunPy-compatible), len matches images
    """
    def __init__(self, images_path: str, headers_path: str, transform=None):
        self.images = np.load(images_path, allow_pickle=True)
        self.headers = np.load(headers_path, allow_pickle=True)
        self.transform = transform
        assert len(self.images) == len(self.headers), "images and headers must have same length"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        hdr = self.headers[idx]

        # If HxWx1, squeeze channel
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        if self.transform is not None:
            img_t = self.transform(img)  # torch tensor CxHxW in [-1,1]
        else:
            # Fallback: ensure CxHxW
            img_t = torch.from_numpy(img.astype(np.float32))
            if img_t.ndim == 2:
                img_t = img_t.unsqueeze(0)
            elif img_t.ndim == 3 and img_t.shape[-1] == 1:
                img_t = img_t.permute(2, 0, 1)

        return img_t, hdr


def collate_keep_headers(batch):
    """
    Batch is a list of (img_tensor CxHxW, header_dict).
    Return:
      imgs: torch.Tensor [B, C, H, W]
      headers: list[dict] of length B
    """
    imgs, headers = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)
    return imgs, list(headers)


# -----------------------
# WandB Logger (owns panel/figure logic)
# -----------------------
class WandBLogger:
    def __init__(
        self,
        enabled: bool,
        project: str = "solar-autoencoder",
        config: dict | None = None,
        run_name: str | None = None,
        watch_model: bool = True,
        watch_log_freq: int = 100,
        # figure/panel defaults
        clip_interval_recon=(1, 100)*u.percent,
        clip_interval_orig=(1, 100)*u.percent,
        dpi: int = 150,
        max_panels_default: int = 4,
        figsize=(12, 5),
    ):
        self.enabled = bool(enabled and WANDB_AVAILABLE)
        self.run = None
        if self.enabled:
            self.run = wandb.init(project=project, config=config or {}, name=run_name)
        else:
            class _NoOp:
                def finish(self_inner): pass
            self.run = _NoOp()

        self._watch_requested = watch_model
        self._watch_log_freq = watch_log_freq
        self._is_watching = False

        # Store panel/figure defaults
        self._clip_interval_recon = clip_interval_recon
        self._clip_interval_orig = clip_interval_orig
        self._dpi = dpi
        self._max_panels_default = max_panels_default
        self._figsize = figsize

    # -------- Public API: metrics, images, artifacts --------
    def maybe_watch(self, model):
        if not self.enabled:
            return
        if self._watch_requested and not self._is_watching:
            try:
                wandb.watch(model, log="all", log_freq=self._watch_log_freq)
                self._is_watching = True
            except Exception as e:
                print(f"[wandb] watch failed: {e}")

    def log_metrics(self, metrics: dict, step: int | None = None):
        if not self.enabled:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[wandb] log metrics failed: {e}")

    def log_image(self, tag: str, image_array: "np.ndarray", step: int | None = None, caption: str | None = None):
        if not self.enabled or image_array is None:
            return
        try:
            img = wandb.Image(image_array, caption=caption)
            wandb.log({tag: img}, step=step)
        except Exception as e:
            print(f"[wandb] log image failed: {e}")

    def log_checkpoint(self, file_path: str, artifact_name: str = "model", artifact_type: str = "model"):
        if not self.enabled:
            return
        try:
            art = wandb.Artifact(artifact_name, type=artifact_type)
            art.add_file(file_path)
            wandb.log_artifact(art)
        except Exception as e:
            print(f"[wandb] log artifact failed: {e}")

    def finish(self):
        try:
            self.run.finish()
        except Exception:
            pass

    # -------- Panel → figure → image internals --------
    @torch.no_grad()
    def _to_uint8_rgb(self, fig, dpi: int | None = None) -> np.ndarray:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi or self._dpi, bbox_inches="tight")
        buf.seek(0)
        img = plt.imread(buf)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        buf.close()
        return img

    @torch.no_grad()
    def _tensor_to_lin_for_sunpy(self, t_single: torch.Tensor) -> np.ndarray:
        """
        Convert a single image tensor in [-1,1] (shape CxHxW or 1xHxW) to
        linear space numpy array for SunPy plotting:
          [-1,1] -> [0,255] -> log space -> inverse log -> linear
        """
        if t_single.ndim == 3:
            if t_single.shape[0] == 1:
                img_chw = t_single
            else:
                img_chw = t_single.mean(dim=0, keepdim=True)
        elif t_single.ndim == 2:
            img_chw = t_single.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {t_single.shape}")

        # [-1,1] -> [0,255]
        x_255 = denormalize_m11_to_255(img_chw)  # 1xHxW
        x_255_hw = x_255[0]  # HxW

        # [0,255] -> log space
        x_log = denormalize_from_0_255_to_log_space(x_255_hw)

        # inverse log -> linear
        x_lin = inverse_log_transform(x_log)  # HxW, float32
        return x_lin

    @torch.no_grad()
    def _render_pair_figure(
        self,
        recon_lin: np.ndarray,
        orig_lin: np.ndarray,
        header: dict,
        suptitle: str | None = None,
        figsize=None,
        clip_interval_recon=None,
        clip_interval_orig=None,
        dpi: int | None = None,
    ) -> tuple:
        recon_map = sunpy.map.Map(recon_lin, header)
        orig_map = sunpy.map.Map(orig_lin, header)

        fig = plt.figure(figsize=figsize or self._figsize)
        ax_left = fig.add_subplot(121, projection=orig_map)
        ax_right = fig.add_subplot(122, projection=recon_map)

        im_orig = orig_map.plot(axes=ax_left, clip_interval=clip_interval_orig or self._clip_interval_orig)
        plt.colorbar(im_orig, ax=ax_left)

        im_recon = recon_map.plot(axes=ax_right, clip_interval=clip_interval_recon or self._clip_interval_recon)
        plt.colorbar(im_recon, ax=ax_right)

        ax_left.set_title("Original")
        ax_right.set_title("Reconstruction")

        t_obs = _format_t_obs(header.get("T_OBS"))
        if suptitle and t_obs:
            fig.suptitle(f"{suptitle}\nObserved: {t_obs}", y=0.98)
        elif suptitle:
            fig.suptitle(suptitle, y=0.98)
        elif t_obs:
            fig.suptitle(f"Observed: {t_obs}", y=0.98)

        fig.tight_layout()

        img_arr = self._to_uint8_rgb(fig, dpi=dpi or self._dpi)
        plt.close(fig)
        return fig, (ax_left, ax_right), img_arr

    @torch.no_grad()
    def _build_panel_stack(
        self,
        preds: torch.Tensor,     # [B, C, H, W] in [-1,1]
        targets: torch.Tensor,   # [B, C, H, W] in [-1,1]
        headers: list[dict],
        max_panels: int = 4,
        suptitle: str | None = None,
        figsize=None,
        clip_interval_recon=None,
        clip_interval_orig=None,
        dpi: int | None = None,
    ) -> np.ndarray | None:
        if preds.ndim != 4 or targets.ndim != 4:
            raise ValueError("preds/targets must be 4D tensors [B, C, H, W]")

        k = min(len(headers), preds.size(0), max_panels)
        panels = []
        for i in range(k):
            hdr = headers[i]
            recon_lin = self._tensor_to_lin_for_sunpy(preds[i].cpu())
            orig_lin = self._tensor_to_lin_for_sunpy(targets[i].cpu())

            _, _, img_arr = self._render_pair_figure(
                recon_lin=recon_lin,
                orig_lin=orig_lin,
                header=hdr,
                suptitle=suptitle,
                figsize=figsize,
                clip_interval_recon=clip_interval_recon,
                clip_interval_orig=clip_interval_orig,
                dpi=dpi,
            )
            panels.append(img_arr)

        if not panels:
            return None

        # Align sizes and stack vertically
        h = min(p.shape[0] for p in panels)
        w = min(p.shape[1] for p in panels)
        resized = [p[:h, :w] for p in panels]
        out = np.concatenate(resized, axis=0)
        return out

    @torch.no_grad()
    def log_recon_panels(
        self,
        tag: str,
        preds: torch.Tensor,      # [B, C, H, W] in [-1,1]
        targets: torch.Tensor,    # [B, C, H, W] in [-1,1]
        headers: list[dict],
        step: int | None = None,
        caption: str | None = None,
        max_panels: int | None = None,
        suptitle: str | None = None,
        figsize=None,
        clip_interval_recon=None,
        clip_interval_orig=None,
        dpi: int | None = None,
    ):
        if not self.enabled:
            return
        try:
            panel_img = self._build_panel_stack(
                preds=preds,
                targets=targets,
                headers=headers,
                max_panels=max_panels or self._max_panels_default,
                suptitle=suptitle,
                figsize=figsize,
                clip_interval_recon=clip_interval_recon,
                clip_interval_orig=clip_interval_orig,
                dpi=dpi,
            )
            if panel_img is None:
                return
            self.log_image(tag=tag, image_array=panel_img, step=step, caption=caption)
        except Exception as e:
            print(f"[wandb] log_recon_panels failed: {e}")


# -----------------------
# Simple Convolutional Autoencoder
# -----------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),     # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),    # H/8
            nn.ReLU(inplace=True),
        )
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 128, 1)

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_ch, 4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1,1]
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bottleneck(z)
        x_hat = self.dec(z)
        return x_hat


# -----------------------
# Training example (can be toggled)
# -----------------------
def main():
    # 1) Device (can be overridden via env)
    device_env = os.getenv("DEVICE")
    device = device_env if device_env else ("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Data (configurable via env)
    images_path = os.getenv("DATA_IMAGES_PATH", "/content/drive/MyDrive/wind_data/images.npy")
    headers_path = os.getenv("DATA_HEADERS_PATH", "/content/drive/MyDrive/wind_data/headers.npy")

    # Training hyperparameters
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    lr = float(os.getenv("LR", "1e-3"))
    epochs = int(os.getenv("EPOCHS", "3"))
    num_workers = int(os.getenv("NUM_WORKERS", "2"))
    pin_memory_env = os.getenv("PIN_MEMORY")
    pin_memory = (pin_memory_env.strip().lower() in ("1","true","yes","y","on")) if pin_memory_env is not None else (device == "cuda")

    dataset = SolarWindDataset(
        images_path=images_path,
        headers_path=headers_path,
        transform=solar_transform,  # yields [1, H, W] in [-1,1]
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_keep_headers,
    )

    # 3) Model/optim on device
    model = ConvAutoencoder(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 4) Logger
    wandb_enabled = os.getenv("WANDB_ENABLED", "true").strip().lower() in ("1","true","yes","y","on")
    wandb_project = os.getenv("WANDB_PROJECT", "solar-ae-demo")
    wandb_run_name = os.getenv("WANDB_RUN_NAME", f"simple-ae-{device}")
    wb = WandBLogger(
        enabled=wandb_enabled,  # set False to disable logging
        project=wandb_project,
        config={"batch_size": batch_size, "lr": lr, "device": device},
        run_name=wandb_run_name,
    )
    wb.maybe_watch(model)

    # 5) Train (minimal example)
    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (x, hdrs) in enumerate(loader):
            x = x.to(device, non_blocking=(device == "cuda"))  # [B, 1, H, W] in [-1,1]
            x_hat = model(x)
            loss = F.mse_loss(x_hat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1

            # Scalar logs
            wb.log_metrics({
                "train/loss": float(loss.detach().cpu()),
                "epoch": epoch,
            }, step=global_step)

            # Periodic SunPy panels: move a small slice to CPU for plotting
            if global_step % 200 == 0 or (epoch == 1 and batch_idx == 0):
                with torch.no_grad():
                    x_vis = x[:4].detach().to("cpu")
                    x_hat_vis = x_hat[:4].detach().to("cpu")
                    hdrs_vis = hdrs[:4]
                    wb.log_recon_panels(
                        tag="ae/recon_panels",
                        preds=x_hat_vis,
                        targets=x_vis,
                        headers=hdrs_vis,
                        step=global_step,
                        caption=f"AE Epoch {epoch}, step {global_step}",
                        suptitle=f"AE Epoch {epoch} Step {global_step}",
                    )

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/autoencoder_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")
    wb.log_checkpoint(ckpt_path, artifact_name="autoencoder_final", artifact_type="model")

    wb.finish()
    print(f"Finished AE demo run on device={device}.")


if __name__ == "__main__":
    # Set these paths before running:
    # images_path: NumPy array (N, H, W) or (N, H, W, 1) with 0..255-like values
    # headers_path: NumPy array (N,) of dicts, each a SunPy-compatible header
    main()