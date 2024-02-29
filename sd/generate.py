import gc
import datetime
import os
import re
from typing import Literal
import subprocess
subprocess.check_call(["pip", "uninstall", "-y", "opencv-python"])
import streamlit as st
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image
)


PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img"]
MODEL_VERSIONS = Literal["2.0", "2.1", "XL 1.0", "XL 1.0 refiner", "Turbo"]


@st.cache_resource(max_entries=1)
def get_pipeline(
    name: PIPELINE_NAMES,
    version: MODEL_VERSIONS = "2.1",
    enable_cpu_offload=False,
) -> DiffusionPipeline:
    
    pipe = None
    if  torch.cuda.is_available():
        torch_type = torch.float16
    else:
        torch_type = torch.float32

    if name == "txt2img" and version == "XL 1.0":
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch_type,
            use_safetensors=True,
            variant="fp16",
        )
        # Potential speedup, but didn't work super great for me.
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    elif name == "img2img" and version == "XL 1.0 refiner":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch_type,
            use_safetensors=True,
            variant="fp16",
        )
    elif name in ["txt2img", "img2img"] and version == "2.1":
        model_id = "stabilityai/stable-diffusion-2-1"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch_type,
        )

        if name == "img2img":
            pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
    elif name == "inpaint" and version == "2.0":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch_type,
        )
    elif name == "inpaint" and version == "XL 1.0":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch_type,
            variant="fp16",
            use_safetensors=True,
        )
    elif name == "txt2img" and version == "Turbo":
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-Turbo", 
            torch_dtype=torch_type, 
            variant="fp16"
        )

    elif name == "img2img" and version == "Turbo":
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-Turbo", 
            torch_dtype=torch_type, 
            variant="fp16"
        )


    if pipe is None:
        raise Exception(f"Could not find pipeline {name} and version {version}")

    if  not torch.cuda.is_available():
        pipe.torch_dtype = torch.float32
    elif enable_cpu_offload:
        print("Enabling CPU offload for pipeline")
        # If we're reeeally strapped for memory, the sequential cpu offload can be used.
        # pipe.enable_sequential_cpu_offload()
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    return pipe


def generate(
    prompt,
    pipeline_name: PIPELINE_NAMES,
    image_input=None,
    mask_input=None,
    negative_prompt=None,
    steps=50,
    width=768,
    height=768,
    guidance_scale=7.5,
    enable_attention_slicing=False,
    enable_cpu_offload=False,
    version="2.1",
    strength=1.0,
    num_images=1
):
    """Generates an image based on the given prompt and pipeline name"""
    prompt = [prompt]*num_images
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)
    callback = lambda step, *_: p.progress(step / steps)

    # NOTE: This code is not being used, since the combined XL 1.0 and refiner
    # pipeline did not work on my hardware.
    refiner_version = None
    try:
        version, refiner_version = version.split(" + ")
    except:
        pass

    pipe = get_pipeline(
        pipeline_name, version=version, enable_cpu_offload=enable_cpu_offload
    )
    if torch.cuda.is_available:
        torch.cuda.empty_cache()

    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        callback=callback,
        guidance_scale=guidance_scale,
    )

    print("kwargs", kwargs)

    if pipeline_name == "inpaint" and image_input and mask_input:
        kwargs.update(image=image_input, mask_image=mask_input, strength=strength)
    elif pipeline_name == "txt2img":
        kwargs.update(width=width, height=height)
    elif pipeline_name == "img2img" and image_input:
        kwargs.update(
            image=image_input,
        )
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    high_noise_frac = 0.8

    # When a refiner is being used, we need to add the denoising_end/start parameters.
    # NOTE: This code is not being used, since the combined XL 1.0 and refiner
    # pipeline did not work on my hardware.
    if refiner_version:
        images = pipe(
            denoising_end=high_noise_frac, output_type="latent", **kwargs
        ).images
    else:
        images = pipe(**kwargs).images

    if refiner_version:
        images = images.detach().clone()
        pipe = None
        gc.collect()
        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        refiner = get_pipeline(
            pipeline_name,
            version=refiner_version,
            enable_cpu_offload=enable_cpu_offload,
        )
        kwargs.pop("width", None)
        kwargs.pop("height", None)
        images = refiner(image=images, denoising_start=high_noise_frac, **kwargs).images
    
    

    os.makedirs("outputs", exist_ok=True)

    filename = (
        "outputs/" + f"_{datetime.datetime.now().timestamp()}"
    )
    for i in range(len(images)):
        images[i].save(f"{filename}_{i}.png")
    with open(f"{filename}_{i}.txt", "w") as f:
        f.write(f"Prompt: {prompt[0]}\n\nNegative Prompt: {negative_prompt}")
    return images
