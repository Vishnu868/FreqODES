"""Stage 1: train the source DeepLabv3 (ResNet18) on CHAOS T1-DUAL in-phase.

The public ODES repo ships no training script or checkpoint, so this trains the
same seg_model used by demo_ODES.py. Usage:
    python train_source.py --src ../data/CHAOS/T1DUAL_InPhase --out checkpoint/CHAOS_T1_INPHASE.pt
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from dataloader.data_loader_stack import Stack_Loader
from model.ODES_deeplab_v3 import seg_model
from utils.helper_functions import add_background_label

TARGET_LABEL = [63, 126, 189, 252]
NUM_CLASSES = 5


def collect_slices(path):
    sets = [Stack_Loader(path, TARGET_LABEL, split=s) for s in ('train', 'test')]
    imgs, labels = [], []
    for ds in ConcatDataset(sets):
        img, label, _ = ds
        lab = torch.argmax(add_background_label(label.float()), dim=1)
        imgs.append(img.float())
        labels.append(lab)
    return torch.cat(imgs), torch.cat(labels)


def augment(x, y):
    if random.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = random.randint(0, 3)
    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    # brightness / contrast jitter
    x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)).clamp(0, 1)
    return x, y


def dice_loss(logits, y):
    p = logits.softmax(1)
    oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[:-1].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count())
    X, Y = collect_slices(args.src)
    print('source slices:', X.shape, 'class counts:', torch.bincount(Y.flatten(), minlength=NUM_CLASSES).tolist())

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = seg_model(num_class=NUM_CLASSES).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0
        for i in range(0, len(X), args.bs):
            idx = perm[i:i + args.bs]
            if len(idx) < 2:
                continue
            xb, yb = [], []
            for j in idx:
                a, b = augment(X[j:j + 1], Y[j:j + 1])
                xb.append(a); yb.append(b)
            xb = torch.cat(xb).unsqueeze(1).repeat(1, 3, 1, 1)
            yb = torch.cat(yb).to(dev)
            xb = xb.to(dev)
            out = model(xb)
            loss = F.cross_entropy(out, yb) + dice_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        print(f'epoch {ep + 1}/{args.epochs} loss {tot / len(X):.4f}', flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)


if __name__ == '__main__':
    main()
