from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset

import torch
from tqdm.auto import tqdm
from utils.vector import getNoise

class CLIP():
    
    # load and initialize CLIP
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder").to('cuda')
    
    # create embeddings from text
    def embed(self, text, pooled=False):
        tokens = self.tokenize(text)
        return self.encode(tokens, pooled)

    # create tokens from text
    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

    # create embeddings from tokens
    def encode(self, tokens, pooled=False):
        with torch.no_grad():
            layers = self.text_encoder(tokens.input_ids.to("cuda"))
        return layers.pooler_output if pooled else layers[0]

    def getEmpty(self, batch_size):
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return self.text_encoder(uncond_input.input_ids.to('cuda'))[0]
        

class UNET():
    def __init__(self):
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
        self.scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.clip = CLIP()
        
        self.unet.to('cuda')
        self.num_inference_steps = 25
        self.guidance_scale = 7.5
        self.max_length = 77
        
    def itterate(self, text_embeddings):

        batch_size = len(text_embeddings)
        uncond_embeddings = self.clip.getEmpty(batch_size)
        text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = getNoise(batch_size)
        latents = latents.to('cuda')
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] *2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
        
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cfg).sample
        
            noise_pred_uncond, noise_pred_text =noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

class VAE():
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
        self.vae.to('cuda')
    def d(self, latent):
        latents_scaled = 1 / 0.18215 * latent
        with torch.no_grad():
            self.vae.enable_slicing()
            image = self.vae.decode(latents_scaled).sample
        processor = VaeImageProcessor()
        return processor.postprocess(image)









        
