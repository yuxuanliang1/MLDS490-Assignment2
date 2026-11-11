"""Execution guard and auditable single-GPU identity."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import torch


def gpu_audit():
    host = socket.gethostname().split('.')[0]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if host != 'mlds-deepdish5':
        raise RuntimeError(f'Training/experiments require mlds-deepdish5, got {host}')
    if not visible or ',' in visible or visible == '-1':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one idle GPU UUID BEFORE starting Python')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one usable CUDA device is required')
    torch.cuda.set_device(0)
    x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
    result = (x @ x.T).sum().item()
    torch.cuda.synchronize()
    if result != 3680.0:
        raise RuntimeError(f'CUDA matrix verification failed: {result}')
    return {
        'hostname': host, 'cwd': str(Path.cwd()), 'python': sys.version,
        'platform': platform.platform(), 'cuda_visible_devices': visible,
        'logical_device': 0, 'device_name': torch.cuda.get_device_name(0),
        'capability': list(torch.cuda.get_device_capability(0)),
        'torch_cuda': torch.version.cuda, 'arch_list': torch.cuda.get_arch_list(),
        'cuda_matmul_sum': result,
        'versions': {p: importlib.metadata.version(p) for p in
                     ('torch', 'gymnasium', 'numpy', 'matplotlib', 'pandas')},
        'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
        'gpu_inventory': subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,uuid,compute_cap,memory.used,utilization.gpu', '--format=csv'], text=True),
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='gpu_audit.json')
    args = parser.parse_args()
    audit = gpu_audit()
    write_json(args.output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
