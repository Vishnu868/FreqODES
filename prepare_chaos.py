"""Convert CHAOS MR T1-DUAL (InPhase / OutPhase) into the stack format read by
dataloader/data_loader_stack.py:   <root>/<Phase>/{train,test}/{data,labels}/<pid>.tif
Each .tif is a (D, H, W) volume; labels keep the CHAOS grey values 63/126/189/252.

    python prepare_chaos.py --chaos ../data_raw/CHAOS_Train_Sets/Train_Sets/MR --out ../data/CHAOS
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pydicom
import tifffile


def to_label(png):
    lab = np.zeros_like(png, dtype=np.uint8)
    for lo, hi, v in [(55, 70, 63), (110, 135, 126), (175, 200, 189), (240, 255, 252)]:
        lab[(png >= lo) & (png <= hi)] = v
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chaos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_test', type=int, default=4)
    args = ap.parse_args()

    pids = sorted(os.listdir(args.chaos), key=int)
    for k, pid in enumerate(pids):
        split = 'test' if k >= len(pids) - args.n_test else 'train'
        base = os.path.join(args.chaos, pid, 'T1DUAL')
        gt_files = sorted(glob.glob(os.path.join(base, 'Ground', '*.png')))
        label = np.stack([to_label(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in gt_files])
        for phase in ('InPhase', 'OutPhase'):
            dcm_files = sorted(glob.glob(os.path.join(base, 'DICOM_anon', phase, '*.dcm')))
            vol = np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in dcm_files])
            assert vol.shape == label.shape, (pid, phase, vol.shape, label.shape)
            for sub, arr in (('data', vol), ('labels', label)):
                d = os.path.join(args.out, f'T1DUAL_{phase}', split, sub)
                os.makedirs(d, exist_ok=True)
                tifffile.imwrite(os.path.join(d, f'{int(pid):03d}.tif'), arr)
        print(pid, split, vol.shape, np.unique(label).tolist())


if __name__ == '__main__':
    main()
