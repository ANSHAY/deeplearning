# %% [markdown]
# # Understanding pipelines, models and schedulers

# %% [markdown]
# 🧨 Diffusers is designed to be a user-friendly and flexible toolbox for building diffusion systems tailored to your use-case. At the core of the toolbox are models and schedulers. While the [DiffusionPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline) bundles these components together for convenience, you can also unbundle the pipeline and use the models and schedulers separately to create new diffusion systems.
# 
# In this tutorial, you'll learn how to use models and schedulers to assemble a diffusion system for inference, starting with a basic pipeline and then progressing to the Stable Diffusion pipeline.

# %% [markdown]
# ## Deconstruct a basic pipeline

# %% [markdown]
# A pipeline is a quick and easy way to run a model for inference, requiring no more than four lines of code to generate an image:

# %%
# from diffusers import DDPMPipeline

# ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
# image = ddpm(num_inference_steps=25).images[0]
# image

# %% [markdown]
# <div class="flex justify-center">
#     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ddpm-cat.png" alt="Image of cat created from DDPMPipeline"/>
# </div>
# 
# That was super easy, but how did the pipeline do that? Let's breakdown the pipeline and take a look at what's happening under the hood.
# 
# In the example above, the pipeline contains a [UNet2DModel](https://huggingface.co/docs/diffusers/main/en/api/models/unet2d#diffusers.UNet2DModel) model and a [DDPMScheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddpm#diffusers.DDPMScheduler). The pipeline denoises an image by taking random noise the size of the desired output and passing it through the model several times. At each timestep, the model predicts the *noise residual* and the scheduler uses it to predict a less noisy image. The pipeline repeats this process until it reaches the end of the specified number of inference steps.
# 
# To recreate the pipeline with the model and scheduler separately, let's write our own denoising process.
# 
# 1. Load the model and scheduler:

# %%
# from diffusers import DDPMScheduler, UNet2DModel

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
# model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")

# %% [markdown]
# 2. Set the number of timesteps to run the denoising process for:

# %%
# scheduler.set_timesteps(50)

# %% [markdown]
# 3. Setting the scheduler timesteps creates a tensor with evenly spaced elements in it, 50 in this example. Each element corresponds to a timestep at which the model denoises an image. When you create the denoising loop later, you'll iterate over this tensor to denoise an image:

# %%
# scheduler.timesteps

# %% [markdown]
# 4. Create some random noise with the same shape as the desired output:

# %%
# import torch

# sample_size = model.config.sample_size
# noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")

# %% [markdown]
# 5. Now write a loop to iterate over the timesteps. At each timestep, the model does a [UNet2DModel.forward()](https://huggingface.co/docs/diffusers/main/en/api/models/unet2d#diffusers.UNet2DModel.forward) pass and returns the noisy residual. The scheduler's [step()](https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddpm#diffusers.DDPMScheduler.step) method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep. This output becomes the next input to the model in the denoising loop, and it'll repeat until it reaches the end of the `timesteps` array.

# %%
# input = noise

# for t in scheduler.timesteps:
#     with torch.no_grad():
#         noisy_residual = model(input, t).sample
#     previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
#     input = previous_noisy_sample

# %% [markdown]
# This is the entire denoising process, and you can use this same pattern to write any diffusion system.
# 
# 6. The last step is to convert the denoised output into an image:

# # %%
# from PIL import Image
# import numpy as np

# image = (input / 2 + 0.5).clamp(0, 1)
# image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
# image = Image.fromarray((image * 255).round().astype("uint8"))
# image

# %% [markdown]
# In the next section, you'll put your skills to the test and breakdown the more complex Stable Diffusion pipeline. The steps are more or less the same. You'll initialize the necessary components, and set the number of timesteps to create a `timestep` array. The `timestep` array is used in the denoising loop, and for each element in this array, the model predicts a less noisy image. The denoising loop iterates over the `timestep`'s, and at each timestep, it outputs a noisy residual and the scheduler uses it to predict a less noisy image at the previous timestep. This process is repeated until you reach the end of the `timestep` array.
# 
# Let's try it out!

# %% [markdown]
# ## Deconstruct the Stable Diffusion pipeline

# %% [markdown]
# Stable Diffusion is a text-to-image *latent diffusion* model. It is called a latent diffusion model because it works with a lower-dimensional representation of the image instead of the actual pixel space, which makes it more memory efficient. The encoder compresses the image into a smaller representation, and a decoder to convert the compressed representation back into an image. For text-to-image models, you'll need a tokenizer and an encoder to generate text embeddings. From the previous example, you already know you need a UNet model and a scheduler.
# 
# As you can see, this is already more complex than the DDPM pipeline which only contains a UNet model. The Stable Diffusion model has three separate pretrained models.
# 
# <Tip>
# 
# 💡 Read the [How does Stable Diffusion work?](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) blog for more details about how the VAE, UNet, and text encoder models.
# 
# </Tip>
# 
# Now that you know what you need for the Stable Diffusion pipeline, load all these components with the [from_pretrained()](https://huggingface.co/docs/diffusers/main/en/api/models/overview#diffusers.ModelMixin.from_pretrained) method. You can find them in the pretrained [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) checkpoint, and each component is stored in a separate subfolder:

# %%
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# %% [markdown]
# Instead of the default [PNDMScheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/pndm#diffusers.PNDMScheduler), exchange it for the [UniPCMultistepScheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler) to see how easy it is to plug a different scheduler in:

# %%
from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# %% [markdown]
# To speed up inference, move the models to a GPU since, unlike the scheduler, they have trainable weights:

# %%
# torch_device = "cpu"
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# %% [markdown]
# ### Create text embeddings

# %% [markdown]
# The next step is to tokenize the text to generate embeddings. The text is used to condition the UNet model and steer the diffusion process towards something that resembles the input prompt.
# 
# <Tip>
# 
# 💡 The `guidance_scale` parameter determines how much weight should be given to the prompt when generating an image.
# 
# </Tip>
# 
# Feel free to choose any prompt you like if you want to generate something else!

# %%
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
batch_size = len(prompt)

# %% [markdown]
# Tokenize the text and generate the embeddings from the prompt:

# %%
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# %% [markdown]
# You'll also need to generate the *unconditional text embeddings* which are the embeddings for the padding token. These need to have the same shape (`batch_size` and `seq_length`) as the conditional `text_embeddings`:

# %%
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

# %% [markdown]
# Let's concatenate the conditional and unconditional embeddings into a batch to avoid doing two forward passes:

# %%
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# %% [markdown]
# ### Create random noise

# %% [markdown]
# Next, generate some initial random noise as a starting point for the diffusion process. This is the latent representation of the image, and it'll be gradually denoised. At this point, the `latent` image is smaller than the final image size but that's okay though because the model will transform it into the final 512x512 image dimensions later.
# 
# <Tip>
# 
# 💡 The height and width are divided by 8 because the `vae` model has 3 down-sampling layers. You can check by running the following:

# %% [markdown]
# 2 ** (len(vae.config.block_out_channels) - 1) == 8

# %% [markdown]
# </Tip>

# %%
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

# %% [markdown]
# ### Denoise the image

# %% [markdown]
# Start by scaling the input with the initial noise distribution, *sigma*, the noise scale value, which is required for improved schedulers like [UniPCMultistepScheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler):

# %%
latents = latents * scheduler.init_noise_sigma

# %% [markdown]
# The last step is to create the denoising loop that'll progressively transform the pure noise in `latents` to an image described by your prompt. Remember, the denoising loop needs to do three things:
# 
# 1. Set the scheduler's timesteps to use during denoising.
# 2. Iterate over the timesteps.
# 3. At each timestep, call the UNet model to predict the noise residual and pass it to the scheduler to compute the previous noisy sample.

# %%
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# %% [markdown]
# ### Decode the image

# %% [markdown]
# The final step is to use the `vae` to decode the latent representation into an image and get the decoded output with `sample`:

# %%
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# %% [markdown]
# Lastly, convert the image to a `PIL.Image` to see your generated image!

# %%
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]

# %% [markdown]
# <div class="flex justify-center">
#     <img src="https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_k_lms.png"/>
# </div>

# %% [markdown]
# ## Next steps

# %% [markdown]
# From basic to complex pipelines, you've seen that all you really need to write your own diffusion system is a denoising loop. The loop should set the scheduler's timesteps, iterate over them, and alternate between calling the UNet model to predict the noise residual and passing it to the scheduler to compute the previous noisy sample.
# 
# This is really what 🧨 Diffusers is designed for: to make it intuitive and easy to write your own diffusion system using models and schedulers.
# 
# For your next steps, feel free to:
# 
# * Learn how to [build and contribute a pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline) to 🧨 Diffusers. We can't wait and see what you'll come up with!
# * Explore [existing pipelines](https://huggingface.co/docs/diffusers/main/en/using-diffusers/../api/pipelines/overview) in the library, and see if you can deconstruct and build a pipeline from scratch using the models and schedulers separately.


