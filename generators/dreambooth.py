import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from transformers import CLIPTextModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def list_folder_names(directory):
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folder_names

class DreamBoothGenerator:
    def __init__(
        self,
        use_cpu_offload=False,
        label = "BEAUFORT"
    ):
        self.label = label
        self.target_directory = "/users/eleves-a/2022/yassine.laraki/Desktop/INF473VV/Cheese-Challenge/generators/saves"
        self.labels = list_folder_names(self.target_directory)
        self.num_inference_steps = 4
        self.guidance_scale = 0

        unet = UNet2DConditionModel.from_pretrained(f"{self.target_directory}/{label}/unet")
        text_encoder = CLIPTextModel.from_pretrained(f"{self.target_directory}/{label}/text_encoder")
        self.pipelines = DiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
                    ).to("cuda")
        self.pipelines.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipelines.scheduler.config, timestep_spacing="trailing"
                    )

    def generate(self, prompts):
        images = self.pipelines(
                prompts,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images

        return images
