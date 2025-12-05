# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union
import json
import numpy as np
import torch
import math
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
)
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from longcat_image.utils.model_utils import split_quotation, prepare_pos_ids, calculate_shift, retrieve_timesteps, optimized_scale
from longcat_image.models.longcat_image_dit import LongCatImageTransformer2DModel
from longcat_image.pipelines.pipeline_output import LongCatImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    width = int(width)
    height = int(height)

    return width, height


class LongCatImageEditPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor", "text_processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        text_processor: AutoProcessor,
        transformer=LongCatImageTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            text_processor=text_processor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.image_processor_vl = text_processor.image_processor
        
        self.image_token = "<|image_pad|>"
        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        self.prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        self.prompt_template_encode_start_idx = 67
        self.prompt_template_encode_end_idx = 5
        self.default_sample_size = 128
        self.max_tokenizer_len = 512
        self.latent_channels = 16


    @torch.inference_mode()
    def encode_prompt(self,
                      image,
                      prompts,
                      device,
                      dtype):
        raw_vl_input = self.image_processor_vl(images=image,return_tensors="pt")
        pixel_values = raw_vl_input['pixel_values']
        image_grid_thw = raw_vl_input['image_grid_thw']

        prompts = [prompt.strip('"') if prompt.startswith('"') and prompt.endswith('"') else prompt for prompt in prompts]
        all_tokens = []
        
        for clean_prompt_sub, matched in split_quotation(prompts[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)['input_ids']
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)

        all_tokens = all_tokens[:self.max_tokenizer_len]
        text_tokens_and_mask = self.tokenizer.pad(
            {'input_ids': [all_tokens]},
            max_length=self.max_tokenizer_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')
        text = self.prompt_template_encode_prefix
        
        merge_length = self.image_processor_vl.merge_size**2
        while self.image_token in text:
            num_image_tokens = image_grid_thw.prod() // merge_length
            text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", self.image_token)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)['input_ids']
        prefix_tokens_mask = torch.tensor([1]*len(prefix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1]*len(suffix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
        
        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1 )
        attention_mask = torch.cat((prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1)

        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw =image_grid_thw,
            output_hidden_states=True
        )
        # [max_sequence_length, batch, hidden_size] -> [batch, max_sequence_length, hidden_size]
        # clone to have a contiguous tensor
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:,self.prompt_template_encode_start_idx: -self.prompt_template_encode_end_idx ,:]

        text_ids = prepare_pos_ids(modality_id=0,
                                   type='text',
                                   start=(0, 0),
                                   num_token=prompt_embeds.shape[1]).to(device, dtype=dtype)

        return prompt_embeds, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        prompt_embeds_length,
        device,
        generator,
        latents=None,
    ): 
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        image_latents, image_latents_ids = None, None

        if image is not None:
            image = image.to(device=self.device, dtype=dtype)
            image_latents = self.vae.encode(image).latent_dist
            image_latents = image_latents.mode()
            image_latents = (image_latents - self.vae.config.shift_factor) *  self.vae.config.scaling_factor
            image_latents = image_latents.to(device=self.device, dtype=dtype)
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, height, width
            )

            image_latents_ids = prepare_pos_ids(modality_id=2,
                                           type='image',
                                           start=(prompt_embeds_length,
                                                  prompt_embeds_length),
                                           height=height//2,
                                           width=width//2).to(device, dtype=torch.float64)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents_ids = prepare_pos_ids(modality_id=1,
                                        type='image',
                                        start=(prompt_embeds_length,
                                               prompt_embeds_length),
                                        height=height//2,
                                        width=width//2).to(device, dtype=torch.float64)

        return latents, image_latents, latents_ids, image_latents_ids


    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0]*1.0/image_size[1])
        
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device

        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = self.image_processor.resize(image, calculated_height//2, calculated_width//2)
        image = self.image_processor.preprocess(image, calculated_height, calculated_width)
        
        negative_prompt = '' if negative_prompt is None else negative_prompt
        negative_prompt = [negative_prompt]*num_images_per_prompt
        prompt = [prompt]*num_images_per_prompt

        prompt_embeds, text_ids = self.encode_prompt(
            image=prompt_image,
            prompts=prompt,
            device=device,
            dtype=torch.float64
        )
        negative_prompt_embeds, negative_text_ids = self.encode_prompt(
            image=prompt_image,
            prompts=negative_prompt, 
            device = device, 
            dtype=torch.float64
        )

        # 4. Prepare latent variables
        num_channels_latents = 16
        latents, image_latents, latents_ids, image_latents_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            calculated_height,
            calculated_width,
            prompt_embeds.dtype,
            prompt_embeds.shape[1],
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
        else:
            prompt_embeds = prompt_embeds.to(device)

        if image is not None:
            latent_image_ids = torch.cat([latents_ids, image_latents_ids], dim=0)
        else:
            latent_image_ids = latents_ids

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                latent_model_input = torch.cat([latent_model_input] * 2) if self.do_classifier_free_guidance else latent_model_input

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :image_seq_len]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, calculated_height, calculated_width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return LongCatImagePipelineOutput(images=image)
