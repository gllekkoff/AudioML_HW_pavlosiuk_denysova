import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import joblib

warnings.filterwarnings("ignore")

np.random.seed(1337)
torch.manual_seed(1337)

DATA_DIR   = Path("/kaggle/input/competitions/birdclef-2026")
MODELS_DIR = Path("/kaggle/input/datasets/gllekk/birdceaf-models")
OUT_PATH   = Path("/kaggle/working/submission.csv")

W_LR     = 1/3
W_CNN    = 1/3
W_EFFNET = 1/3

SR         = 32_000
DURATION   = 5
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 1_024
CLIP_LEN   = SR * DURATION
MFCC_N     = 40
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

print(f"Using device: {DEVICE}")

class_labels = pd.read_csv(DATA_DIR / "taxonomy.csv")["primary_label"].tolist()
n_classes    = len(class_labels)
print(f"Classes: {n_classes}")

_mel_transform  = None
_mfcc_transform = None


def get_mel_transform():
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = T.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        ).to(DEVICE)
    return _mel_transform


def get_mfcc_transform():
    global _mfcc_transform
    if _mfcc_transform is None:
        _mfcc_transform = T.MFCC(
            sample_rate=SR,
            n_mfcc=MFCC_N,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
        ).to(DEVICE)
    return _mfcc_transform


def load_audio_as_tensor(path: str) -> torch.Tensor:
    """Load with torchaudio (faster than librosa) and resample on GPU."""
    sig, rate = torchaudio.load(path)
    if sig.shape[0] > 1:
        sig = sig.mean(0, keepdim=True)
    sig = sig.to(DEVICE)
    if rate != SR:
        sig = torchaudio.functional.resample(sig, rate, SR)
    return sig  # (1, n_samples)


def chunkify(wav: torch.Tensor) -> torch.Tensor:
    """Split into 5-second chunks and stack into a single batch tensor."""
    step   = CLIP_LEN
    total  = wav.shape[1]
    chunks = []
    for offset in range(0, max(total, step), step):
        chunk = wav[:, offset: offset + step]
        if chunk.shape[1] < step:
            pad   = torch.zeros(1, step - chunk.shape[1], device=DEVICE)
            chunk = torch.cat([chunk, pad], dim=1)
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)  # (n_chunks, CLIP_LEN)


def batch_mel_specs(chunks_t: torch.Tensor) -> torch.Tensor:
    """Compute mel spectrograms for all chunks in one GPU call.

    Returns: (n_chunks, 1, n_mels, time)  — ready for CNN / EfficientNet.
    """
    mel = get_mel_transform()(chunks_t)
    mel = torch.clamp(mel, min=1e-5).log10()
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std  = mel.std(dim=(1, 2), keepdim=True)
    mel  = (mel - mean) / (std + 1e-6)
    return mel.unsqueeze(1)


def batch_mfcc_features(chunks_t: torch.Tensor) -> np.ndarray:
    """Compute MFCC mean+std for all chunks in one GPU call.

    Returns: (n_chunks, 2*MFCC_N) numpy array on CPU.
    """
    mfcc = get_mfcc_transform()(chunks_t)        # (n, n_mfcc, time)
    feat = torch.cat([mfcc.mean(dim=2), mfcc.std(dim=2)], dim=1)  # (n, 2*n_mfcc)
    return feat.cpu().numpy().astype(np.float32)


def predict_nn(model: nn.Module, mel_batch: torch.Tensor) -> np.ndarray:
    """Run model inference in batches. mel_batch: (n, 1, n_mels, time) on GPU."""
    all_preds = []
    with torch.no_grad():
        for start in range(0, mel_batch.shape[0], BATCH_SIZE):
            out = torch.sigmoid(model(mel_batch[start: start + BATCH_SIZE]))
            all_preds.append(out.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


class SimpleCNN(nn.Module):
    def __init__(self, n_cls: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_efficientnet(n_cls: int) -> nn.Module:
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    old = model.features[0][0]
    new = nn.Conv2d(
        1, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=False,
    )
    new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    model.features[0][0] = new
    model.classifier[1]  = nn.Linear(model.classifier[1].in_features, n_cls)
    return model


lr_pipeline = joblib.load(MODELS_DIR / "lr_pipeline.pkl")
print("Loaded LR pipeline.")

cnn_model = SimpleCNN(n_classes).to(DEVICE)
cnn_model.load_state_dict(torch.load(MODELS_DIR / "cnn_model.pt", map_location=DEVICE))
cnn_model.eval()
print("Loaded SimpleCNN.")

effnet_model = build_efficientnet(n_classes).to(DEVICE)
effnet_model.load_state_dict(torch.load(MODELS_DIR / "effnet_model.pt", map_location=DEVICE))
effnet_model.eval()
print("Loaded EfficientNet-B0.")

test_soundscape_path = DATA_DIR / "test_soundscapes"
test_soundscapes = sorted(
    str(test_soundscape_path / f)
    for f in os.listdir(test_soundscape_path)
    if f.endswith(".ogg")
)
print(f"Test soundscapes found: {len(test_soundscapes)}")

row_ids     = []
scores_list = []

for soundscape in test_soundscapes:
    try:
        wav      = load_audio_as_tensor(soundscape)
        chunks_t = chunkify(wav)
        stem     = Path(soundscape).stem

        mel_batch  = batch_mel_specs(chunks_t)
        feats_mfcc = batch_mfcc_features(chunks_t)

        preds_lr  = lr_pipeline.predict_proba(feats_mfcc).astype(np.float32)
        preds_cnn = predict_nn(cnn_model,    mel_batch)
        preds_eff = predict_nn(effnet_model, mel_batch)

        preds_ens = W_LR * preds_lr + W_CNN * preds_cnn + W_EFFNET * preds_eff

        n = chunks_t.shape[0]
        row_ids.extend(f"{stem}_{(i + 1) * DURATION}" for i in range(n))
        scores_list.append(preds_ens)

        print(f"{stem} — {n} chunks done.")

    except Exception as e:
        import traceback
        print(f"ERROR on {soundscape}: {e}")
        traceback.print_exc()

probs = np.concatenate(scores_list, axis=0).astype(np.float32)

submission = pd.DataFrame(probs, columns=class_labels)
submission.insert(0, "row_id", row_ids)

assert submission.columns.tolist() == ["row_id"] + class_labels
assert not submission.isna().any().any()

submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}  shape={submission.shape}")
print(submission.iloc[:3, :8])
