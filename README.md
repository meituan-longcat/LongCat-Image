# LongCat-Image

<div align="center">
  <img src="assets/longcat-image_logo.svg" width="45%" alt="LongCat-Image" />
</div>
<hr>

<div align="center" style="line-height: 1;">
    <a href='https://arxiv.org/pdf/2512.07584'><img src='https://img.shields.io/badge/Technical Report-PDF-red'></a>
    <a href='https://longcat.ai/'><img src="https://img.shields.io/badge/🤖%20Demo-LongCat--Image-ADFF2F?color=29E154&logoColor=white"></a>
    <a href='https://github.com/meituan-longcat/LongCat-Image'><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
    <a href='https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png'><img src='https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white'></a>
    <a href='https://x.com/Meituan_LongCat'><img src='https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white'></a>
</div>

<div align="center" style="line-height: 1;">

[//]: # (  <a href='https://meituan-longcat.github.io/LongCat-Image/'><img src='https://img.shields.io/badge/Project-Page-green'></a>)
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image-blue'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image-Dev'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image--Dev-blue'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image-Edit'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image--Edit-blue'></a>
</div>

## Model Introduction
We introduce **LongCat-Image**, a pioneering open-source and bilingual (Chinese-English) foundation
model for image generation, designed to address core challenges in multilingual text rendering,
photorealism, deployment efficiency, and developer accessibility prevalent in current leading models.


<div align="center">
  <img src="assets/model_struct.jpg" width="90%" alt="LongCat-Image Model Architecture" />
</div>


### Key Features
- 🌟 **Exceptional Efficiency and Performance**: With only **6B parameters**, LongCat-Image surpasses numerous open-source models that are several times larger across multiple benchmarks, demonstrating the immense potential of efficient model design.
- 🌟 **Superior Editing Performance**: LongCat-Image-Edit model achieves state-of-the-art performance among open-source models, delivering leading instruction-following and image quality with superior visual consistency.
- 🌟 **Powerful Chinese Text Rendering**: LongCat-Image demonstrates superior accuracy and stability in rendering common Chinese characters compared to existing SOTA open-source models and achieves industry-leading coverage of the Chinese dictionary.
- 🌟 **Remarkable Photorealism**: Through an innovative data strategy and training framework, LongCat-Image achieves remarkable photorealism in generated images.
- 🌟 **Comprehensive Open-Source Ecosystem**: We provide a complete toolchain, from intermediate checkpoints to full training code, significantly lowering the barrier for further research and development.

[//]: # (For more details, please refer to the comprehensive [***LongCat-Image Technical Report***]&#40;https://arxiv.org/abs/2412.11963&#41;.)

## 🎨 Showcase

### Text-to-Image

<div align="center">
  <img src="assets/gallery.jpeg" width="90%" alt="LongCat-Image Generation Examples" />
</div>

### Image Editing

<div align="center">
  <img src="assets/image_edit_gallery.jpg" width="90%" alt="LongCat-Image-Edit Generation Examples" />
</div>

## Quick Start

### Installation

Clone the repo:

```shell
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Image
cd LongCat-Image
```

Install dependencies:

```shell
# create conda environment
conda create -n longcat-image python=3.10
conda activate longcat-image

# install other requirements
pip install -r requirements.txt
python setup.py develop
```

### Model Download

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Models</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Type</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Description</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Download Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text&#8209;to&#8209;Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Final Release. The standard model for out&#8209;of&#8209;the&#8209;box inference.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image">Huggingface</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image&#8209;Dev</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text&#8209;to&#8209;Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Development. Mid-training checkpoint, suitable for fine-tuning.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image-Dev">Huggingface</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image&#8209;Edit</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Specialized model for image editing.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image-Edit">Huggingface</a></span>
        </td>
      </tr>
    </tbody>
  </table>
</div>



Download models using huggingface-cli:
```shell
pip install "huggingface_hub[cli]"
huggingface-cli download meituan-longcat/LongCat-Image --local-dir ./weights/LongCat-Image
huggingface-cli download meituan-longcat/LongCat-Image-Dev --local-dir ./weights/LongCat-Image-Dev
huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir ./weights/LongCat-Image-Edit
```

### Run Text-to-Image Generation
> [!TIP]
> Leveraging a stronger LLM for prompt refinement can further enhance image generation quality. Please refer to [inference_t2i.py](https://github.com/meituan-longcat/LongCat-Image/blob/main/scripts/inference_t2i.py#L28) for detailed usage instructions.

> [!CAUTION]
> **Special Handling for Text Rendering**
>
> For both Text-to-Image and Image Editing tasks involving text generation, **you must enclose the target text within quotes (`""`)**.
>
> **Reason:** The tokenizer applies **character-level encoding** specifically to content found inside quotes. Failure to use explicit quotation marks will result in a significant degradation of text rendering quality.

```python
import torch
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline

device = torch.device('cuda')
checkpoint_dir = './weights/LongCat-Image'

text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
    torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

pipe = LongCatImagePipeline.from_pretrained(
    checkpoint_dir,
    transformer=transformer,
    text_processor=text_processor,
    torch_dtype=torch.bfloat16
)
# pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~17 GB); slower but prevents OOM

prompt = '一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。镜头采用中距离视角，突出她的神态和服饰的细节。光线柔和地打在她的脸上，强调她的五官和饰品的质感，增加画面的层次感与亲和力。整个画面构图简洁，砖墙的纹理与阳光的光影效果相得益彰，突显出人物的优雅与从容。'

image = pipe(
    prompt,
    height=768,
    width=1344,
    guidance_scale=4.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    generator=torch.Generator("cpu").manual_seed(43),
    enable_cfg_renorm=True,
    enable_prompt_rewrite=True  # Reusing the text encoder as a built-in prompt rewriter
).images[0]
image.save('./t2i_example.png')
```

### Run Image Editing

```python
import torch
from PIL import Image
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline

device = torch.device('cuda')
checkpoint_dir = './weights/LongCat-Image-Edit'
text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
    torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

pipe = LongCatImageEditPipeline.from_pretrained(
    checkpoint_dir,
    transformer=transformer,
    text_processor=text_processor,
    torch_dtype=torch.bfloat16
)
# pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~19 GB); slower but prevents OOM

generator = torch.Generator("cpu").manual_seed(43)
img = Image.open('assets/test.png').convert('RGB')
prompt = '将猫变成狗'
image = pipe(
    img,
    prompt,
    negative_prompt='',
    guidance_scale=4.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    generator=generator
).images[0]

image.save('./edit_example.png')

```

## Evaluation Results

### Text-to-Image Generation

The quantitative evaluation results on public benchmarks demonstrate LongCat-Image's competitive performance:

| Model                   | Accessibility | Parameters | GenEval↑ | DPG↑      | WISE↑    |
|-------------------------|---------------|:----------:|:--------:|:--------:|:--------:|
| FLUX.1&#8209;dev        | Open Source   | 12B        | 0.66     | 83.84     | 0.50     |
| GPT Image 1 [High]      | Proprietary   | -          | 0.84     | 85.15     | -        |
| HunyuanImage&#8209;3.0  | Open Source   | 80B        | 0.72     | 86.10     | 0.57     |
| Qwen&#8209;Image        | Open Source   | 20B        | **0.87** | **88.32** | 0.62     |
| Seedream 4.0            | Proprietary   | -          | 0.84     | 88.25     | **0.78** |
| **LongCat&#8209;Image** | Open Source   | **6B**     | **0.87** | 86.80     | 0.65     |

### Text Rendering

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" rowspan="2">Model</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" rowspan="2">
          <div align="center">GlyphDraw2↑</div>
        </th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" colspan="3">CVTG‑2K↑</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" rowspan="2">
          <div align="center">ChineseWord↑</div>
        </th>
      </tr>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">
          <div align="center">Acc</div>
        </th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">
          <div align="center">NED</div>
        </th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">
          <div align="center">CLIPScore</div>
        </th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">HunyuanImage&#8209;3.0</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.78</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.7650</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.8765</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><strong>0.8121</strong></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><u>58.5</u></div></td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Qwen&#8209;Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.93</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.8288</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.9297</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><u>0.8059</u></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">56.6</div></td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Seedream&nbsp;4.0</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><b>0.97</b></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><strong>0.8917</strong></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><strong>0.9507</strong></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.7853</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">49.3</div></td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;"><strong>LongCat&#8209;Image</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><u>0.95</u></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><u>0.8658</u></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><u>0.9361</u></div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center">0.7859</div></td>
        <td style="padding: 8px; border: 1px solid #d0d7de;"><div align="center"><b>90.7</b></div></td>
      </tr>
    </tbody>
  </table>
</div>


### Human Evaluation (MOS)

| **Model**         | **Alignment↑** | **Plausibility↑** | **Realism↑** | **Aesthetics↑** |
|:------------------|:----------------:|:-----------------:|:------------:|:---------------:|
| HunyuanImage‑3.0  | 3.40           | 3.33              | 3.50         | 3.04            |
| Qwen‑Image        | 3.95           | 3.48              | 3.45         | 3.09            |
| Seedream&nbsp;4.0  | **4.25**       | **3.76**          | 3.54         | **3.10**        |
| **LongCat‑Image** | 3.99           | 3.48              | **3.60**     | 3.06            |

### Image Editing
#### Performance comparison on CEdit-Bench and GEdit-Bench:

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" rowspan="2">Model</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" colspan="3">CEdit&#8209;Bench&#8209;EN↑</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" colspan="3">CEdit&#8209;Bench&#8209;CN↑</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" colspan="3">GEdit&#8209;Bench&#8209;EN↑</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;" colspan="3">GEdit&#8209;Bench&#8209;CN↑</th>
      </tr>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_SC</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_PQ</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_O</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_SC</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_PQ</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_O</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_SC</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_PQ</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_O</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_SC</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_PQ</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">G_O</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FLUX.1&nbsp;Kontext&nbsp;[Pro]</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.79</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.80</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.53</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.15</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.07</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.43</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.02</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.60</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.56</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.11</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.36</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.23</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">GPT&nbsp;Image&nbsp;1&nbsp;[High]</td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.64</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.26</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.17</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.67</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.26</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.21</b></td><td style="padding: 8px; border: 1px solid #d0d7de;">7.85</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.62</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.53</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.67</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.56</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.30</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Nano&nbsp;Banana</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.51</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.17</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.20</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.67</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.21</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.36</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.86</td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.33</b></td><td style="padding: 8px; border: 1px solid #d0d7de;">7.54</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.51</td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.31</b></td><td style="padding: 8px; border: 1px solid #d0d7de;">7.25</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border-bottom: 3px solid #d0d7de;">Seedream&nbsp;4.0</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">8.12</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">7.95</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">7.58</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">8.14</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">7.95</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">7.57</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;"><b>8.24</b></td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">8.08</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;"><b>7.68</b></td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;"><b>8.19</b></td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;">8.14</td><td style="padding: 8px; border-bottom: 3px solid #d0d7de;"><b>7.71</b></td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FLUX.1&nbsp;Kontext&nbsp;[Dev]</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.31</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.56</td><td style="padding: 8px; border: 1px solid #d0d7de;">5.93</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.25</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.66</td><td style="padding: 8px; border: 1px solid #d0d7de;">1.51</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.52</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.38</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.00</td><td style="padding: 8px; border: 1px solid #d0d7de;">-</td><td style="padding: 8px; border: 1px solid #d0d7de;">-</td><td style="padding: 8px; border: 1px solid #d0d7de;">-</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Step1X&#8209;Edit</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.68</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.36</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.25</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.88</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.28</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.35</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.66</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.35</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.97</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.20</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.87</td><td style="padding: 8px; border: 1px solid #d0d7de;">6.86</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Qwen&#8209;Image&#8209;Edit</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.07</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.84</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.52</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.03</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.78</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.46</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.00</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.86</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.56</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.82</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.79</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.52</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Qwen&#8209;Image&#8209;Edit&nbsp;[2509]</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.04</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.79</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.48</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.93</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.71</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.37</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.15</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.86</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.54</td><td style="padding: 8px; border: 1px solid #d0d7de;">8.05</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.88</td><td style="padding: 8px; border: 1px solid #d0d7de;">7.49</td></tr>
      <tr><td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image&#8209;Edit</td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.27</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.88</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.67</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.25</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.85</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.65</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.18</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.00</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.64</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>8.08</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.99</b></td><td style="padding: 8px; border: 1px solid #d0d7de;"><b>7.60</b></td></tr>
    </tbody>
  </table>
</div>


#### Human Evaluation (Win Rate)

| **Models** | **Comprehensive Quality** | **Consistency** |
|-----------|:-------------------------:|:-----------------:|
| Nano Banana vs LongCat-Image-Edit | **60.8%** vs 39.2% | **53.9%** vs 46.1% | 
| Seedream 4.0 vs LongCat-Image-Edit | **56.9%** vs 43.1% | **56.3%** vs 43.7% |
| Qwen-Image-Edit [2509] vs LongCat-Image-Edit | 41.3% vs **58.7%** | 45.8% vs **54.2%** |
| FLUX.1 Kontext [Pro] vs LongCat-Image-Edit | 39.5% vs **60.5%** | 37% vs **63%** |

## Training Pipeline

We provide training code that enables advanced development of our LongCat‑Image‑Dev and model, including SFT, LoRA, DPO, and Image Editing training.

See [TRAINING.md](./train_examples/README.md) for detailed instructions.

## Community Works

Community works are welcome! Please PR or inform us in Issue to add your work.

- [LoRA Adapters] Fine-tuned models for specific styles and domains
- [ComfyUI Integration] Native support for ComfyUI workflow
- [Diffusers Pipeline] HuggingFace Diffusers integration
- [ComfyUI Longcat Image](https://github.com/sooxt98/comfyui_longcat_image) - Custom node extension for ComfyUI workflow.


## License Agreement

LongCat-Image is licensed under Apache 2.0.
See the [LICENSE](LICENSE) file for the full license text.

## Usage Considerations

This model has not been specifically designed or comprehensively evaluated for every possible downstream application.

Developers should take into account the known limitations of large language models, including performance variations across different languages, and carefully assess accuracy, safety, and fairness before deploying the model in sensitive or high-risk scenarios.
It is the responsibility of developers and downstream users to understand and comply with all applicable laws and regulations relevant to their use case, including but not limited to data protection, privacy, and content safety requirements.

Nothing in this Model Card should be interpreted as altering or restricting the terms of the Apache License 2.0 under which the model is released.

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{LongCat-Image,
      title={LongCat-Image Technical Report},
      author={Meituan LongCat Team and  Hanghang Ma and Haoxian Tan and Jiale Huang and Junqiang Wu and Jun-Yan He and Lishuai Gao and Songlin Xiao and Xiaoming Wei and Xiaoqi Ma and Xunliang Cai and Yayong Guan and Jie Hu},
	    journal={arXiv preprint arXiv:2512.07584},
      year={2025}
}
```

## Acknowledgements

We would like to thank the contributors to the [FLUX.1](https://github.com/black-forest-labs/flux), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.

## Contact

Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or join our WeChat Group if you have any questions.

#### WeChat Group
<img src=assets/wechat_qrcode.png width="200px">

---

<div align="center">
  <sub>Built with ❤️ by Meituan LongCat Team</sub>
</div>
