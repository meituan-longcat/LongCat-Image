
import os

import torch.distributed as dist

def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    if not dist.is_available():
        return 0
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank():
    if not dist.is_available():
        return 0
    return int(os.getenv('LOCAL_RANK', 0)) if dist.is_initialized() else 0


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0
