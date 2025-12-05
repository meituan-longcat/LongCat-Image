from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import numpy as np
import copy
from torch.utils.data import IterableDataset

from longcat_image.utils.dist_utils import get_world_size, get_rank, get_local_rank

class MultiResolutionDistributedSampler(torch.utils.data.Sampler):
    def __init__(self,
                 batch_size: int,
                 dataset: IterableDataset,
                 data_resolution_infos: List,
                 bucket_info: dict,
                 num_replicas: int = None,
                 rank: int = None,
                 seed: int = 888,
                 epoch: int = 0,
                 shuffle: bool = True):

        if not dist.is_available():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = get_world_size()
            rank = get_rank()

        self.len_items = len(dataset)
        bucket_info = {float(b): bucket_info[b] for b in bucket_info.keys()}
        self.aspect_ratios = np.array(sorted(list(bucket_info.keys())))
        self.resolutions = np.array([bucket_info[aspect] for aspect in self.aspect_ratios])

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch
        self.shuffle = shuffle
        self.seed = seed
        self.cur_rank_index = []
        self.rng = np.random.RandomState(seed+self.epoch)
        self.global_batch_size = batch_size*num_replicas
        self.data_resolution_infos = np.array(data_resolution_infos, dtype=np.float32)
        print(f'num_replicas {num_replicas}, cur rank {rank}!!!')

        self.split_to_buckets()
        self.num_samples = len(dataset)//num_replicas

    def split_to_buckets(self):
        self.buckets = {}
        self._buckets_bak = {}
        data_aspect_ratio = self.data_resolution_infos[:,0]*1.0/self.data_resolution_infos[:, 1]
        bucket_id = np.abs(data_aspect_ratio[:, None] - self.aspect_ratios).argmin(axis=1)
        for i in range(len(self.aspect_ratios)):
            self.buckets[i] = np.where(bucket_id == i)[0]
            self._buckets_bak[i] = np.where(bucket_id == i)[0]
        for k, v in self.buckets.items():
            print(f'bucket {k}, resolutions {self.resolutions[k]}, sampler nums {len(v)}!!!')

    def get_batch_index(self):
        success_flag = False
        while not success_flag:
            bucket_ids = list(self.buckets.keys())
            bucket_probs = [len(self.buckets[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            chosen_id = int(self.rng.choice(bucket_ids, 1, p=bucket_probs)[0])
            if len(self.buckets[chosen_id]) < self.global_batch_size:
                del self.buckets[chosen_id]
                continue
            batch_data = self.buckets[chosen_id][:self.global_batch_size]
            batch_data = (batch_data, self.resolutions[chosen_id])
            self.buckets[chosen_id] = self.buckets[chosen_id][self.global_batch_size:]
            if len(self.buckets[chosen_id]) == 0:
                del self.buckets[chosen_id]
            success_flag = True
            assert bool(self.buckets), 'There is not enough data in the current epoch.'
        return batch_data

    def shuffle_bucker_index(self):
        self.rng = np.random.RandomState(self.seed+self.epoch)
        self.buckets = copy.deepcopy(self._buckets_bak)
        for bucket_id in self.buckets.keys():
            self.rng.shuffle(self.buckets[bucket_id])

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if len(self.cur_rank_index) == 0:
                global_batch_index, target_resolutions = self.get_batch_index()
                self.cur_rank_index = list(map(
                    int, global_batch_index[self.batch_size*self.rank:self.batch_size*(self.rank+1)]))
                self.resolution = list(map(int, target_resolutions))
            data_index = self.cur_rank_index.pop(0)
            return (data_index, self.resolution)

        except Exception as e:
            self.epoch += 1
            self.shuffle_bucker_index()
            print(f'get error {e}.')
            raise StopIteration

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
