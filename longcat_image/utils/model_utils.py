from typing import Any, Callable, Dict, List, Optional, Union

import re
import math
from PIL import Image

import torch
import torch.nn as nn
import inspect
from transformers import AutoTokenizer, AutoProcessor


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels //
                              (2 * 2), height * 2, width * 2)

    return latents


def split_quotation(prompt, quote_pairs=None):
    """
    Implement a regex-based string splitting algorithm that identifies delimiters defined by single or double quote pairs. 

    Examples::
        >>> prompt_en = "Please write 'Hello' on the blackboard for me."
        >>> print(split_quotation(prompt_en))
        >>> # output: [('Please write ', False), ("'Hello'", True), (' on the blackboard for me.', False)]
    """
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []

    for i, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = 'longcat_$##$_longcat' * (i+1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ('‘', '’'), ('“', '”')]
        quotes = ["'", '"', '‘', '’', '“', '”']
        for q1 in quotes:
            for q2 in quotes:
                if (q1, q2) not in quote_pairs:
                    quote_pairs.append((q1, q2))

    pattern = '|'.join([re.escape(q1) + r'[^' + re.escape(q1+q2) +
                       r']*?' + re.escape(q2) for q1, q2 in quote_pairs])

    parts = re.split(f'({pattern})', prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


def encode_prompt(prompt: str, tokenizer: AutoTokenizer, text_tokenizer_max_length: int, prompt_template_encode_prefix: str, prompt_template_encode_suffix: str):

    all_tokens = []
    for clean_prompt_sub, matched in split_quotation(prompt):
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = tokenizer(sub_word, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)
        else:
            tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
            all_tokens.extend(tokens)

    all_tokens = all_tokens[:text_tokenizer_max_length]
    text_tokens_and_mask = tokenizer.pad(
        {'input_ids': [all_tokens]},
        max_length=text_tokenizer_max_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')
    
    prefix_tokens = tokenizer(prompt_template_encode_prefix, add_special_tokens=False)['input_ids']
    suffix_tokens = tokenizer(prompt_template_encode_suffix, add_special_tokens=False)['input_ids']
    prefix_tokens_mask = torch.tensor( [1]*len(prefix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )
    suffix_tokens_mask = torch.tensor( [1]*len(suffix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )

    prefix_tokens = torch.tensor(prefix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
    suffix_tokens = torch.tensor(suffix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
    
    input_ids = torch.cat( (prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1 )
    attention_mask = torch.cat( (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1 )


    input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
    attention_mask = text_tokens_and_mask['attention_mask'].squeeze(0)

    return input_ids, attention_mask

def encode_prompt_edit(prompt: str, img: Image.Image,tokenizer: AutoTokenizer, image_processor_vl: AutoProcessor, text_tokenizer_max_length: int, prompt_template_encode_prefix: str, prompt_template_encode_suffix: str):
    raw_vl_input = image_processor_vl(images=img,return_tensors="pt")
    pixel_values = raw_vl_input['pixel_values'].squeeze(0)
    image_grid_thw = raw_vl_input['image_grid_thw'].squeeze(0)

    all_tokens = []
    for clean_prompt_sub, matched in split_quotation(prompt):
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = tokenizer(sub_word, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)
        else:
            tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
            all_tokens.extend(tokens)

    all_tokens = all_tokens[:text_tokenizer_max_length]
    text_tokens_and_mask = tokenizer.pad(
        {'input_ids': [all_tokens]},
        max_length=text_tokenizer_max_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')
    
    text = prompt_template_encode_prefix
    merge_length = image_processor_vl.merge_size**2
    while  "<|image_pad|>" in text:
        num_image_tokens = image_grid_thw.prod() // merge_length
        text = text.replace( "<|image_pad|>", "<|placeholder|>" * num_image_tokens, 1)
    text = text.replace("<|placeholder|>",  "<|image_pad|>")
    
    prefix_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    suffix_tokens = tokenizer(prompt_template_encode_suffix, add_special_tokens=False)['input_ids']
    prefix_tokens_mask = torch.tensor( [1]*len(prefix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )
    suffix_tokens_mask = torch.tensor( [1]*len(suffix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )

    prefix_tokens = torch.tensor(prefix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
    suffix_tokens = torch.tensor(suffix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
    
    input_ids = torch.cat( (prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1 )
    attention_mask = torch.cat( (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1 )


    # input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
    # attention_mask = text_tokens_and_mask['attention_mask'].squeeze(0)

    return input_ids, attention_mask, pixel_values, image_grid_thw


def prepare_pos_ids(
        modality_id=0,
        type='text',
        start=(0, 0),
        num_token=None,
        height=None,
        width=None):
    if type == 'text':
        assert num_token
        if height or width:
            print(
                'Warning: The parameters of height and width will be ignored in "text" type.')
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == 'image':
        assert height and width
        if num_token:
            print('Warning: The parameter of num_token will be ignored in "image" type.')
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = (
            pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        )
        pos_ids[..., 2] = (
            pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        )
        pos_ids = pos_ids.reshape(height*width, 3)
    else:
        raise KeyError(f'Unknow type {type}, only support "text" or "image".')
    # pos_ids = pos_ids[None, :].repeat(batch_size, 1, 1)
    return pos_ids


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(
            scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# @torch.cuda.amp.autocast(dtype=torch.float32)
@torch.amp.autocast('cuda', dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star