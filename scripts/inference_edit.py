import torch
from PIL import Image
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline

if __name__ == '__main__':

    device = torch.device('cuda')
    checkpoint_dir = './weights/LongCat-Image-Edit'
    
    text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
    transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
        torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

    pipe = LongCatImageEditPipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16,
    )
    # pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~19 GB); slower but prevents OOM

    generator = torch.Generator("cpu").manual_seed(43)
    img = Image.open('assets/test.png').convert('RGB')
    prompt = '将猫变成狗'
    
    cfg = 2.5 # original 4.5
    steps = 30 # original 50
    
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=cfg,
        num_inference_steps=steps,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    image.save('./edit_example.png')
