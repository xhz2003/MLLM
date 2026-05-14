#!/usr/bin/env python3
# measure_model_size_time.py
# Usage examples:
# python measure_model_size_time.py --ckpt models/ckpt_200.pth --dataset VLFDataset_h5/Harvard_test.h5
# python measure_model_size_time.py --ckpt models/ckpt_200.pth --dataset VLFDataset_h5/Harvard_test.h5 --iters 100 --warmup 10

import os
import time
import argparse
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# --- 修改为你项目中的实际导入路径 ---
from net.Film import Net
from utils.H5_read import H5ImageTextDataset
from utils.img_read_save import img_save   # optional, not used for timing

def get_num_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def get_param_size_mb(num_params, bytes_per_param=4):
    # assume float32 (4 bytes)
    return num_params * bytes_per_param / (1024 ** 2)

def get_state_dict_file_size_mb(model, tmp_dir=None):
    # Save state_dict to temporary file to measure actual size on disk
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "tmp_model_state_dict.pth")
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_mb

def measure_inference_time(model, dataloader, device,
                           warmup=10, iters=100, max_images=None, use_cuda_sync=True):
    """
    Measures average inference time per forward pass (per batch).
    - warmup: number of warmup iterations to ignore (for CUDA caches)
    - iters: number of measured iterations (if dataloader shorter, will loop)
    - max_images: optional cap on total images to measure
    Returns average_time_per_image (seconds), avg_time_per_batch (seconds), total_images_measured
    """
    model.eval()
    total_time = 0.0
    measured_iters = 0
    total_images = 0

    # create an iterator that loops over dataset if needed
    data_iter = iter(dataloader)
    # Warm-up
    for _ in range(warmup):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_IR, data_VIS, text, idx = batch
        data_IR = torch.FloatTensor(data_IR).to(device)
        data_VIS = torch.FloatTensor(data_VIS).to(device)
        text = text.squeeze(1).to(device)
        with torch.no_grad():
            if device.type == 'cuda' and use_cuda_sync:
                torch.cuda.synchronize()
                _ = model(data_IR, data_VIS, text)
                torch.cuda.synchronize()
            else:
                _ = model(data_IR, data_VIS, text)

    # Timed iterations
    for i in range(iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_IR, data_VIS, text, idx = batch
        bsz = data_IR.shape[0]
        if max_images is not None and total_images >= max_images:
            break

        data_IR = torch.FloatTensor(data_IR).to(device)
        data_VIS = torch.FloatTensor(data_VIS).to(device)
        text = text.squeeze(1).to(device)

        with torch.no_grad():
            if device.type == 'cuda' and use_cuda_sync:
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model(data_IR, data_VIS, text)
                torch.cuda.synchronize()
                t1 = time.time()
            else:
                t0 = time.time()
                _ = model(data_IR, data_VIS, text)
                t1 = time.time()

        elapsed = t1 - t0
        total_time += elapsed
        measured_iters += 1
        total_images += bsz

        # optional: early exit if we've measured enough images
        if max_images is not None and total_images >= max_images:
            break

    if measured_iters == 0:
        return float('nan'), float('nan'), 0

    avg_time_per_batch = total_time / measured_iters
    avg_time_per_image = total_time / total_images if total_images > 0 else float('nan')
    return avg_time_per_image, avg_time_per_batch, total_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pth')
    parser.add_argument('--dataset', type=str, required=True, help='Path to test h5 file (or directory) used by H5ImageTextDataset')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=200, help='Number of timed iterations (batches)')
    parser.add_argument('--max_images', type=int, default=None, help='Optional: stop after this many images')
    parser.add_argument('--no_disk_save', action='store_true', help='If set, skip saving state_dict to disk measurement')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # build model (adjust constructor args to match your Net)
    model = Net(hidden_dim=256, image2text_dim=32)
    # If multiple GPUs available, wrap in DataParallel just like your test script
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # load checkpoint
    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device)
    # try common keys
    if 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    # handle single-GPU <-> DataParallel key mismatch
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try stripping/adding 'module.' prefix
        new_state = {}
        if all(k.startswith('module.') for k in state.keys()):
            # strip module.
            for k, v in state.items():
                new_state[k.replace('module.', '', 1)] = v
        else:
            # add module.
            for k, v in state.items():
                new_state['module.' + k] = v
        model.load_state_dict(new_state)

    # dataset & dataloader
    dataset = H5ImageTextDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 1) parameter counts and size estimates ---
    num_params = get_num_parameters(model)
    size_params_mb = get_param_size_mb(num_params)
    print(f"Number of parameters (trainable): {num_params:,}")
    print(f"Estimated parameter memory (float32): {size_params_mb:.3f} MB")

    # 1b) save state_dict to disk and measure actual file size (optional)
    if not args.no_disk_save:
        try:
            disk_mb = get_state_dict_file_size_mb(model)
            print(f"Saved state_dict file size: {disk_mb:.3f} MB (on disk)")
        except Exception as e:
            print("Warning: failed to save state dict for disk size measurement:", e)

    # --- 2) measure inference time ---
    print("Starting timing ...")
    avg_time_per_image, avg_time_per_batch, total_images = measure_inference_time(
        model=model,
        dataloader=dataloader,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        max_images=args.max_images,
        use_cuda_sync=(device.type == 'cuda')
    )

    print(f"Measured on {total_images} images (timed iters: {args.iters}, warmup: {args.warmup})")
    print(f"Average inference time per image: {avg_time_per_image:.6f} seconds")
    print(f"Average inference time per batch: {avg_time_per_batch:.6f} seconds (batch_size={args.batch_size})")
    if device.type == 'cuda':
        # report GPU memory used by the model (approx)
        try:
            mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Peak GPU memory allocated during measurement: {mem_alloc:.3f} MB")
        except Exception:
            pass

if __name__ == '__main__':
    main()
