#!/usr/bin/env python3
"""Train a binary success classifier on D405/D455 images.

Uses a lightweight ResNet-18 (pretrained) to classify task success/failure.
Output: success probability (0-1), used for automated reward labeling.

Usage:
    .venv/bin/python examples/CR5AF/train_success_classifier.py \
        --data-dir /datasets/cr5af_grasp_housing_l50_v2 \
        --output-dir /tmp/cr5af_classifier_v1
"""
import argparse, json, os
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


class SuccessDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.15):
        self.data_dir = Path(data_dir)
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
        import pandas as pd
        parquet_files = sorted(self.data_dir.glob("data/**/*.parquet"))
        self.samples = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            label = 1 if float(df['reward'].iloc[0]) > 0.5 else 0
            T_ep = len(df)
            for t in range(0, T_ep, 5):
                self.samples.append((pq_file, t, label))
        np.random.seed(42)
        idxs = np.random.permutation(len(self.samples))
        n_val = int(len(self.samples) * val_ratio)
        if split == "train":
            self.samples = [self.samples[i] for i in idxs[n_val:]]
        else:
            self.samples = [self.samples[i] for i in idxs[:n_val]]
        print(f"{split}: {len(self.samples)}")

    def _get_frame(self, pq_file, t):
        ep_str = pq_file.stem
        for cam in ["hand_view", "table_view"]:
            chunk_dir = pq_file.parent.parent.parent / "videos" / pq_file.parent.name / \
                       f"observation.images.{cam}"
            candidates = list(chunk_dir.glob(f"{ep_str}.mp4"))
            if candidates:
                cap = cv2.VideoCapture(str(candidates[0]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pq_file, t, label = self.samples[idx]
        img = self._get_frame(pq_file, t)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        for p in backbone.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(self.encoder(x).flatten(1))).squeeze(1)


def train(data_dir, output_dir, epochs=50, batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_ds = SuccessDataset(data_dir, "train")
    val_ds = SuccessDataset(data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size, num_workers=0)
    model = Classifier().to(device)
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        correct, total, losses = 0, 0, []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = bce(preds, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            correct += ((preds > 0.5) == labels.bool()).sum().item()
            total += len(labels)
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += ((model(imgs) > 0.5) == labels.bool()).sum().item()
                v_total += len(labels)
        val_acc = v_correct / max(v_total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"[{ep+1:3d}] loss={np.mean(losses):.4f} "
                  f"train={(correct/total):.3f} val={val_acc:.3f}")
    print(f"Best val_acc: {best_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/datasets/cr5af_grasp_housing_l50_v2")
    parser.add_argument("--output-dir", default="/tmp/cr5af_classifier_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size)
