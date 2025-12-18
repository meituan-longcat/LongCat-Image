import torch
from PIL import Image
from diffusers import LongCatImageEditPipeline

if __name__ == '__main__':
    device = torch.device('cuda')

    pipe = LongCatImageEditPipeline.from_pretrained("meituan-longcat/LongCat-Image-Edit", torch_dtype= torch.bfloat16 )
    # pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~18 GB); slower but prevents OOM

    img = Image.open('assets/test.png').convert('RGB')
    prompt = '将猫变成狗'
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(43)
    ).images[0]

    image.save('./edit_example.png')
