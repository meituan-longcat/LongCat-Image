import torch

from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline
from misc.prompt_rewrite_api import prompt_rewrite_deepseek

if __name__ == '__main__':

    device = torch.device('cuda')
    checkpoint_dir = './weights/LongCat-Image'

    text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
    transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
        torch_dtype=torch.bfloat16, use_safetensors=True).to(device)
    
    pipe = LongCatImagePipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor
    )
    pipe.to(device, torch.bfloat16)

    prompt = '一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。镜头采用中距离视角，突出她的神态和服饰的细节。光线柔和地打在她的脸上，强调她的五官和饰品的质感，增加画面的层次感与亲和力。整个画面构图简洁，砖墙的纹理与阳光的光影效果相得益彰，突显出人物的优雅与从容。'
    
    enable_prompt_rewrite_api = False
    if enable_prompt_rewrite_api:
        prompt = prompt_rewrite_deepseek( prompt )

    image = pipe(
        prompt,
        negative_prompt='',
        height=768,
        width=1344,
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator= torch.Generator("cpu").manual_seed(43),
        enable_cfg_renorm=True,
        enable_prompt_rewrite=True if not enable_prompt_rewrite_api else False
    ).images[0]
    image.save('./t2i_example.png')
