from typing import Any, Callable, Dict, List, Optional, Union
import os
import random
import traceback
import math
import json
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm

from longcat_image.dataset import MULTI_RESOLUTION_MAP
from longcat_image.utils import encode_prompt
from longcat_image.dataset import MultiResolutionDistributedSampler

Image.MAX_IMAGE_PIXELS = 2000000000

MAX_RETRY_NUMS = 100

class DpoPairDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: dict,
                 txt_root: str,
                 tokenizer: AutoTokenizer,
                 resolution: tuple = (1024, 1024)):
        super(DpoPairDataSet, self).__init__()
        self.resolution = resolution
        self.text_tokenizer_max_length = cfg.text_tokenizer_max_length
        self.null_text_ratio = cfg.null_text_ratio
        self.aspect_ratio_type = cfg.aspect_ratio_type
        self.aspect_ratio = MULTI_RESOLUTION_MAP[self.aspect_ratio_type]
        self.tokenizer = tokenizer

        self.total_datas = []
        self.data_resolution_infos = []
        with open(txt_root, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                data = json.loads(line.strip())
                try:
                    height, widht = int(data['height']), int(data['width'])
                    self.data_resolution_infos.append((height, widht))
                    self.total_datas.append(data)
                except Exception as e:
                    print(f'get error {e}, data {data}.')
                    continue
        self.data_nums = len(self.total_datas)

    def transform_img(self, image, original_size, target_size):
        img_h, img_w = original_size
        target_height, target_width = target_size

        original_aspect = img_h / img_w  # height/width
        crop_aspect = target_height / target_width

        if original_aspect >= crop_aspect:
            resize_width = target_width
            resize_height = math.ceil(img_h * (target_width/img_w))
        else:
            resize_width = math.ceil(img_w * (target_height/img_h))
            resize_height = target_height

        image = T.Compose([
            T.Resize((resize_height, resize_width),interpolation=InterpolationMode.BICUBIC),  # Image.LANCZOS
            T.CenterCrop((target_height, target_width)),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])(image)

        return image

    def __getitem__(self, index_tuple):
        index, target_size = index_tuple

        for _ in range(MAX_RETRY_NUMS):
            try:
                item = self.total_datas[index]
                img_path_win = item["img_path_win"]
                img_path_lose = item["img_path_lose"]

                prompt = item['prompt']

                if random.random() < self.null_text_ratio:
                    prompt = ''

                raw_image_win = Image.open(img_path_win).convert('RGB')
                raw_image_lose = Image.open(img_path_lose).convert('RGB')
                assert raw_image_win is not None and raw_image_lose is not None
                img_w, img_h = raw_image_win.size

                raw_image_win = self.transform_img(raw_image_win, original_size=(
                    img_h, img_w), target_size= target_size )
                raw_image_lose = self.transform_img(raw_image_lose, original_size=(
                    img_h, img_w), target_size= target_size )

                input_ids,attention_mask = encode_prompt(prompt, self.tokenizer, self.text_tokenizer_max_length)
                
                return {"image_win": raw_image_win, "image_lose": raw_image_lose,  "prompt": prompt, 'input_ids': input_ids, 'attention_mask': attention_mask}

            except Exception as e:
                traceback.print_exc()
                print(f"failed read data {e}!!!")
                index = random.randint(0, self.data_nums-1)

    def __len__(self):
        return self.data_nums

    def collate_fn(self, batchs):
        images_win = torch.stack([example["image_win"] for example in batchs])
        images_lose = torch.stack([example["image_lose"] for example in batchs])
        input_ids = torch.stack([example["input_ids"] for example in batchs])
        attention_mask = torch.stack([example["attention_mask"] for example in batchs])
        prompts = [example['prompt'] for example in batchs]
        batch_dict = {
            "images_win": images_win,
            "images_lose": images_lose,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
        }
        return batch_dict


def build_dataloader(cfg: dict,
                     csv_root: str,
                     tokenizer: AutoTokenizer,
                     resolution: tuple = (1024, 1024)):
    dataset = DpoPairDataSet(cfg, csv_root, tokenizer, resolution)

    sampler = MultiResolutionDistributedSampler(batch_size=cfg.train_batch_size, dataset=dataset,
                                                data_resolution_infos=dataset.data_resolution_infos,
                                                bucket_info=dataset.aspect_ratio,
                                                epoch=0,
                                                num_replicas=None,
                                                rank=None
                                                )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        sampler=sampler,
        shuffle=None,
    )
    return train_loader


if __name__ == '__main__':
    import sys
    import argparse
    from torchvision.transforms.functional import to_pil_image

    txt_root = 'xxxx'

    cfg = argparse.Namespace(
        txt_root=txt_root,
        text_tokenizer_max_length=256,
        resolution=1024,
        text_encoder_path="xxx",
        center_crop=True,
        dataloader_num_workers=0,
        null_text_ratio=0.1,
        train_batch_size=16,
        seed=0,
        aspect_ratio_type='mar_1024',
        revision=None)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_path, trust_remote_code=True)
    data_loader = build_dataloader(cfg, cfg.csv_root, tokenizer, cfg.resolution)

    _oroot = f'./debug_data_example_show'
    os.makedirs(_oroot, exist_ok=True)

    cnt = 0
    for epoch in range(1):
        print(f"Start, epoch {epoch}!!!")
        for i_batch, batch in enumerate(data_loader):
            print(batch['attention_mask'].shape)
            print(batch['images_win'].shape,'-',batch['images_lose'].shape,)
            
            if cnt > 100:
                break
            cnt += 1
