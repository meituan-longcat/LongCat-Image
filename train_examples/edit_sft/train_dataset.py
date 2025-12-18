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
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from tqdm import tqdm

from longcat_image.dataset import MULTI_RESOLUTION_MAP
from longcat_image.utils import encode_prompt_edit
from longcat_image.dataset import MultiResolutionDistributedSampler

Image.MAX_IMAGE_PIXELS = 2000000000

MAX_RETRY_NUMS = 100

class Text2ImageLoraDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: dict,
                 txt_root: str,
                 tokenizer: AutoTokenizer,
                 text_processor:AutoProcessor,
                 resolution: tuple = (1024, 1024),
                 repeats: int = 1 ):
        super(Text2ImageLoraDataSet, self).__init__()
        self.resolution = resolution
        self.text_tokenizer_max_length = cfg.text_tokenizer_max_length
        self.null_text_ratio = cfg.null_text_ratio
        self.aspect_ratio_type = cfg.aspect_ratio_type
        self.aspect_ratio = MULTI_RESOLUTION_MAP[self.aspect_ratio_type]
        self.tokenizer = tokenizer
        self.image_processor_vl = text_processor.image_processor

        self.prompt_template_encode_prefix = cfg.prompt_template_encode_prefix
        self.prompt_template_encode_suffix = cfg.prompt_template_encode_suffix
        self.prompt_template_encode_start_idx = cfg.prompt_template_encode_start_idx
        self.prompt_template_encode_end_idx = cfg.prompt_template_encode_end_idx

        self.total_datas = []
        self.data_resolution_infos = []
        with open(txt_root, 'r') as f:
            lines = f.readlines()
            lines *= cfg.repeats
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
        print(f'get sampler {len(self.total_datas)}, from {txt_root}!!!')

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
            T.Resize((resize_height, resize_width),interpolation=InterpolationMode.BICUBIC), 
            T.CenterCrop((target_height, target_width)),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])(image)

        return image
    def transform_img_vl(self, image, original_size, target_size):
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
            T.Resize((resize_height, resize_width),interpolation=InterpolationMode.BICUBIC), 
            T.CenterCrop((target_height, target_width)),
            T.Resize((target_height//2, target_width//2)),
        ])(image)

        return image

    def __getitem__(self, index_tuple):
        index, target_size = index_tuple

        for _ in range(MAX_RETRY_NUMS):
            try:
                item = self.total_datas[index]
                img_path = item["img_path"]
                ref_img_path = item["ref_img_path"]
                prompt = item['prompt']

                if random.random() < self.null_text_ratio:
                    prompt = ''

                raw_image = Image.open(img_path).convert('RGB')
                ref_image = Image.open(ref_img_path).convert('RGB')
                assert raw_image is not None
                img_w, img_h = raw_image.size

                ref_image_vl = self.transform_img_vl(ref_image, original_size=(img_h, img_w), target_size= target_size )
                raw_image = self.transform_img(raw_image, original_size=(img_h, img_w), target_size= target_size )
                ref_image = self.transform_img(ref_image, original_size=(img_h, img_w), target_size= target_size )
                
                input_ids, attention_mask, pixel_values, image_grid_thw = encode_prompt_edit(prompt,ref_image_vl, self.tokenizer, self.image_processor_vl,self.text_tokenizer_max_length, self.prompt_template_encode_prefix, self.prompt_template_encode_suffix )
                return {"image": raw_image, "ref_image":ref_image, "prompt": prompt, 'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values':pixel_values, 'image_grid_thw':image_grid_thw}

            except Exception as e:
                traceback.print_exc()
                print(f"failed read data {e}!!!")
                index = random.randint(0, self.data_nums-1)

    def __len__(self):
        return self.data_nums

    def collate_fn(self, batchs):
        images = torch.stack([example["image"] for example in batchs])
        ref_images = torch.stack([example["ref_image"] for example in batchs])
        input_ids = torch.stack([example["input_ids"] for example in batchs])
        attention_mask = torch.stack([example["attention_mask"] for example in batchs])
        pixel_values = torch.stack([example["pixel_values"] for example in batchs])
        image_grid_thw = torch.stack([example["image_grid_thw"] for example in batchs])
        prompts = [example['prompt'] for example in batchs]
        batch_dict = {
            "images": images,
            "ref_images": ref_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
            "pixel_values":pixel_values,
            "image_grid_thw":image_grid_thw
        }
        return batch_dict


def build_dataloader(cfg: dict,
                     csv_root: str,
                     tokenizer: AutoTokenizer,
                     text_processor: AutoProcessor,
                     resolution: tuple = (1024, 1024)):
    dataset = Text2ImageLoraDataSet(cfg, csv_root, tokenizer, text_processor, resolution)

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
