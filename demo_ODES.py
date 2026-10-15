# print('python demo_ODES.py --cfg cfgs/ODES_CHAOST1_IP_OOP.yaml')
print('python demo_ODES.py --cfg cfgs/TTA_Active_CHAOST1_IP_OOP.yaml')
print('python demo_ODES.py --cfg cfgs/TTA_Active_CHAOS_DUKE.yaml')


import argparse
import json
import time
import copy
import logging
import os, pdb
import random

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm


# Configuration
from conf import cfg, load_cfg_fom_args

# Dataloader
from dataloader.data_loader_stack import Stack_Loader
# from dataloader.ODES_data_loader_stack import Stack_Loader

# Method
from methods.ODES import TTA_Active
from methods.freq_shift import FrequencyReference

# Model
from model.ODES_deeplab_v3 import seg_model

# Eval
from utils.CHAOSmetrics import DICE
from utils.helper_functions import *




# Logging setup
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load configuration
load_cfg_fom_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Model will run on {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU"}')
print("DEVICE:", device)


# Utility functions
def plot_result(out_enc, label, img_path, result_dir='results'):
    # os.makedirs(result_dir, exist_ok=True)
    out_pred = torch.argmax(out_enc, dim=1)
    out_gt = torch.argmax(label, dim=1)
    combined = (torch.abs(torch.cat([out_pred, out_gt], dim=1) - 4).cpu().numpy()) * 60
    tifffile.imwrite(img_path, combined.astype(np.uint8))
    # pdb.set_trace()

    # tifffile.imwrite(os.path.join(result_dir, os.path.basename(img_path)), combined.astype(np.uint8))

def Evaluate(pred, gt):
    dice = [DICE(pred[:, i, :, :], gt[:, i, :, :]).cpu().numpy() for i in range(pred.shape[1])]
    return dice

def Jaccard(pred, gt):
    out = []
    for i in range(pred.shape[1]):
        p, g = pred[:, i], gt[:, i]
        inter = (p * g).sum(); union = p.sum() + g.sum() - inter
        out.append(float(inter / (union + 1e-8)))
    return out

def average_value(scores, mode='Dice'):

    avg_scores = [np.mean([score[i] for score in scores]) for i in range(len(organs))]

    for organ, score in zip(organs, avg_scores):
        print(f"{mode} {organ}: {score}")
    
    print(f"Mean {mode} Score:", np.mean(avg_scores))
    return avg_scores

def collect_params(model):
    params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for np, p in m.named_parameters() if np in ['weight', 'bias']]
    return params

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def setup_optimizer(params):
    method = cfg.OPTIM.METHOD.upper()
    lr = cfg.OPTIM.LR

    if method == 'SGD':
        return optim.SGD(params, lr=lr, momentum=cfg.OPTIM.MOMENTUM, weight_decay=cfg.OPTIM.WD, nesterov=cfg.OPTIM.NESTEROV)
    elif method == 'ADAM':
        return optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError(f"Optimizer '{method}' not implemented.")

def data_setup(dataset, index):
    img, label, img_path = dataset[index]
    img = torch.cat([img.unsqueeze(1)] * 3, dim=1).float().to(device)

    if label.shape[1] < cfg.MODEL.NUM_CLASSES - 1:
        label_padded = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES - 1, *label.shape[2:])
        label_padded[:, :label.shape[1], :, :] = label
        label = label_padded

    label = add_background_label(label.float().to(device))
    return img, label, img_path

def get_feature_stat_auto(model):
    mean, var = zip(*[(layer.running_mean.clone(), layer.running_var.clone()) for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)])
    return mean, var

def setup_active_learning(model, pre_mean, pre_var, freq_ref=None):
    model = configure_model(model)
    optimizer = setup_optimizer(collect_params(model))
    return TTA_Active(model, optimizer, pre_mean, pre_var, cfg, freq_ref)


def build_freq_reference():
    """FreqODES: frequency reference from a small random subset of source slices."""
    src = ConcatDataset([Stack_Loader(cfg.DATASET.SOURCE_PATH, target_label=target_label, split=s) for s in ('train', 'test')])
    slices = np.concatenate([src[i][0].numpy() for i in range(len(src))])
    rng = np.random.RandomState(0)
    pick = rng.choice(len(slices), min(cfg.DATASET.FREQ_REF_SLICES, len(slices)), replace=False)
    return FrequencyReference(slices[pick])


# Setup values
if cfg.MODEL.ADAPTATION=='CHAOS':
    target_label = [63, 126, 189, 252]
    # CHAOS MR label values: 63 liver, 126 right kidney, 189 left kidney, 252 spleen
    organs = ['liver', 'right kidney', 'left kidney', 'spleen']
elif cfg.MODEL.ADAPTATION=='DUKE':
    target_label = [1]
    organs = ['liver']
elif cfg.MODEL.ADAPTATION=='PROSTRATE':
    target_label = [1]
    organs = ['liver']



# Data preparation
train_data = Stack_Loader(cfg.DATASET.TARGET_PATH, target_label=target_label, split='train')
test_data = Stack_Loader(cfg.DATASET.TARGET_PATH, target_label=target_label, split='test')
full_test_data = ConcatDataset([train_data, test_data])

# Model initialization
# pdb.set_trace()
base_model = seg_model(num_class=cfg.MODEL.NUM_CLASSES).to(device)
base_model.load_state_dict(torch.load(cfg.DATASET.PRETRAINED_PATH, map_location=device))
torch.set_num_threads(int(os.environ.get('NTHREADS', os.cpu_count())))



pre_mean, pre_var = get_feature_stat_auto(base_model)
freq_ref = build_freq_reference() if cfg.ACTIVE.PRUNE.upper() in ('FFT', 'DWT', 'FREQ', 'FUSED', 'FREQV2', 'FREQAREA', 'FREQENT') else None
if cfg.MODEL.Inference_Only:
    base_model.eval()
else:
    adaptive_model = setup_active_learning(copy.deepcopy(base_model), pre_mean, pre_var, freq_ref)

# Evaluation loop (each patient volume = one incoming batch)
dice_all, per_batch, n_ann, jac_all = [], [], [], []
SAVE_PRED = [int(v) for v in os.environ.get('SAVE_PRED', '').split(',') if v]
preds = {}
t0 = time.time()
for i in tqdm(range(len(full_test_data))):
    img, label, img_path = data_setup(full_test_data, i)
    if cfg.MODEL.Inference_Only:
        with torch.no_grad():
            output = base_model(img)
        n_ann.append(0)
    else:
        _, output = adaptive_model(img, label, 1)
        n_ann.append(adaptive_model.n_annotated)
    oh = get_one_hot_encoding(output.detach(), cfg.MODEL.NUM_CLASSES)
    dice = Evaluate(oh, label)
    jac_all.append(Jaccard(oh, label))
    if i in SAVE_PRED:
        preds[f'img_{i}'] = img[:, 0].cpu().numpy().astype(np.float16)
        preds[f'gt_{i}'] = torch.argmax(label, 1).cpu().numpy().astype(np.uint8)
        preds[f'pred_{i}'] = torch.argmax(output.detach(), 1).cpu().numpy().astype(np.uint8)
    dice_all.append(dice)
    per_batch.append(float(np.mean([float(d) for d in dice[:len(organs)]])))

print("Adapted Model Results:")
avg = average_value(dice_all, mode='Adapt Dice All Data')
total_px = sum(full_test_data[i][0].numel() for i in range(len(full_test_data)))
result = dict(name=cfg.EXP_NAME_RUN if 'EXP_NAME_RUN' in cfg else cfg.EXP_NAME,
              prune=cfg.ACTIVE.PRUNE, K=cfg.ACTIVE.BUDGET_SLICE, b=cfg.ACTIVE.BUDGET_REGION,
              lam=cfg.ACTIVE.FUSE_LAMBDA, div_weight=cfg.ACTIVE.DIV_WEIGHT, source_only=cfg.MODEL.Inference_Only,
              organs=organs, organ_dice=[100 * float(a) for a in avg],
              organ_jaccard=[100 * float(np.mean([j[k] for j in jac_all])) for k in range(len(organs))],
              mean_jaccard=100 * float(np.mean([np.mean(j[:len(organs)]) for j in jac_all])), mean_dice=100 * float(np.mean(avg)),
              per_batch_mean_dice=[100 * d for d in per_batch], annotated_pixels=int(sum(n_ann)),
              annotated_fraction=float(sum(n_ann)) / total_px, seconds=time.time() - t0)
os.makedirs('results/json', exist_ok=True)
tag = cfg.EXP_NAME
if preds:
    os.makedirs('results/pred', exist_ok=True)
    np.savez_compressed(f'results/pred/{tag}.npz', **preds)
with open(f'results/json/{tag}.json', 'w') as f:
    json.dump(result, f, indent=1)
print(json.dumps({k: result[k] for k in ('prune', 'K', 'mean_dice', 'annotated_fraction', 'seconds')}))
