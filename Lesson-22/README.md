# Lesson 22: Image Generation Models

In this lesson, we will explore image generation models and learn how to run them locally.

We'll delve into the Stable Diffusion process used in AI image generation, focusing on how to use these models to generate images from text prompts and existing images.

We'll examine how the transformer architecture, previously used in text generation models, can also be applied to image generation. We'll study the noising and denoising processes and how they've been adapted in recent years to create sophisticated image generation algorithms.

Additionally, we'll set up the Stable Diffusion WebUI application to easily load, manage, and run Image Generation AI models locally.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Python tools installed on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency with `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Familiarity with `venv` for creating and managing virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for cloning repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Basic understanding of Python programming language syntax
  - [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
- Active account on [OpenAI Platform](https://platform.openai.com/)
  - To run API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 21

- Image generation techniques
- Image generation AI models
- Stable diffusion
- Image generation applications
- OpenAI Image Generation API
- Using DALL·E model from OpenAI

## Running Generative AI Models

- Generative AI models
  - Closed models
    - Proprietary models developed by companies like OpenAI, Anthropic, or Google
    - Often have restricted access and usage policies
    - Examples include DALL·E, Midjourney, and Imagen
  - Open Source (or Open Access) models
    - Publicly available models with more flexible usage terms
    - Can be downloaded, modified, and run locally
    - Examples include Stable Diffusion and open-source versions of CLIP
- Running models
  - Local execution: Running models on your own hardware
  - Cloud-based services: Using platforms like Google Colab or AWS to run models
  - API integration: Accessing models through web APIs provided by companies
- Using Hugging Face's [Stable Diffusion Pipelines](https://huggingface.co/docs/diffusers/quicktour)
  - Hardware requirements
    - GPU with at least 10GB VRAM for optimal performance
    - Can run on CPU, but significantly slower
  - Tutorial: [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)
    - Step-by-step guide to using Stable Diffusion in a cloud environment
  - Dependencies
    - Python 3.7+
    - PyTorch 1.7.0+
    - Transformers library
    - Diffusers library
  - [Installation](https://huggingface.co/docs/diffusers/installation)
    - Detailed official instructions for installing the Diffusers library and its dependencies
  - Usage
    - [Loading models and configuring schedulers](https://huggingface.co/docs/diffusers/using-diffusers/loading)
      - How to load pre-trained models and set up different sampling methods
    - [Running a pipeline](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)
      - Steps to execute the image generation process
  - Text-to-image Pipeline [parameters](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
    - Detailed explanation of various parameters that control the image generation process
    - Includes options for prompt, negative prompt, number of inference steps, and guidance scale

### Using Diffusion Models with Python

- Practical exercise
  - Exercise 1: [Generate Images with Stable Diffusion](./exercises/00-Generate-Images-Diffusion.md)
    - Code and run [Diffusion.py](./examples/Diffusion.py) to generate an image from a text prompt

## Setting Up Stable Diffusion WebUI

- The [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) tool
  - Inspired the `Text Generation WebUI` we used previously
  - Built on [Gradio](https://www.gradio.app/) as well
  - Similar interface and usage
- Addressing the [Dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies)
  - Instructions for [NVIDIA](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
  - Instructions for [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
  - Instructions for [Intel](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
  - Instructions for [Apple](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
  - Options for running in [Docker Containers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Containers)
  - Options for running in [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)
- Instructions for [Installing and Running](https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#installation-and-running)
- Installing and running packages with [Stability Matrix](https://github.com/LykosAI/StabilityMatrix)
  - **Note**: You can install Stable Diffusion WebUI using the [Stability Matrix](https://github.com/LykosAI/StabilityMatrix) tool
  - Choose the correct distribution for your OS from the [Downloads page](https://lykos.ai/downloads)
  - Install the latest stable release
  - Run the tool and choose `+ Add Package` in the `Packages` tab
  - Select `Stable Diffusion WebUI` from the list and click `Install`
- Alternative [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services) if your environment is incompatible with the dependencies
- Usage
  - Access the WebUI at <http://127.0.0.1:7860/> (default location)
  - Run with the `--api` flag to access the API at <http://127.0.0.1:7860/docs>
- Key [Features](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features)
  - Stable Diffusion: Loading, managing, and running stable diffusion models
  - Text to Image tasks: Generates images based on textual prompts provided by the user
  - Image to Image tasks: Transforms existing images based on text prompts or other images
  - Inpainting: Allows editing specific parts of an image while keeping the rest intact
  - Outpainting: Extends images beyond their original boundaries, creating seamless expansions
  - Image variations: Generates multiple versions of an image with slight variations
  - Resizing/Upscaling: Increases the resolution and quality of images without losing detail
- Using Models and Checkpoints
- Managing configurations
- Navigating the interface

## Using Model Checkpoints

- Downloading models
  - Using Stability Matrix to manage models
    - Checkpoint manager
    - Model Browser
      - CivitAI
      - Hugging Face
  - Using Stable Diffusion WebUI
    - From [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&library=diffusers&sort=downloads) (filter for "Diffusers" under "Library")
      - You can easily download models using a [model downloader extension](https://github.com/Iyashinouta/sd-model-downloader)
        - Download and install using the `Extensions` tab
          - Use `Install from URL` tab
          - Paste the extension URL in the `URL for extension's git repository` input field
          - Click the `Install` button
          - Go to the `Installed` tab
          - Click the `Apply and restart UI` button
        - After installing, go to the `Model Downloader` tab
    - From [CivitAI](https://civitai.com/models?tag=base+model) (filter for "Checkpoints" under "Model Types")
      - A secure way to download models from CivitAI is to use the [CivitAI Helper Extension](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper)
        - Download and install using the `Extensions` tab
          - Use `Install from URL` tab
          - Paste the extension URL in the `URL for extension's git repository` input field
          - Click the `Install` button
          - Go to the `Installed` tab
          - Click the `Apply and restart UI` button
        - Go to `Civitai Helper` tab
          - Use the `Download Model` pane to get model information from a link and download it
            - You can use the `Block NSFW Level Above` filter to conveniently filter out images tagged as NSFW
    - From other websites like [PromptHero](https://prompthero.com/ai-models/text-to-image) and [Reddit](https://www.reddit.com/r/StableDiffusion/wiki/models/)
- Models
  - High resolution model [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - **Note**: Not suitable for generating readable text, faces, or objects with compositional structure
  - Standard resolution photorealistic model [Photon](https://civitai.com/models/84728/photon)
    - **Note**: Has some issues with rendering hands and fingers
  - Versatile model for digital art, drawings, anime, and painting [GhostMix](https://civitai.com/models/36520?modelVersionId=76907)
    - Requires the use of LoRAs for optimal results
    - **Note**: Not very effective for photorealistic images or generating images of "common" persons or scenes
  - Versatile high resolution model [JuggernautXL](https://civitai.com/models/133005/juggernaut-xl)
    - Good generalist model that effectively utilizes light/shadow effects
- Choosing models
  - Model file extensions
    - `ckpt` files are models where Python can load serialized data, but they are potentially dangerous as they can execute arbitrary code
    - `safetensors` files are not affected by the vulnerability of `ckpt` files and are preferred when available
  - Model weights
    - `EMAonly` models use only the Exponential Moving Average of the weights
      - These models are smaller and faster to process, requiring less VRAM to run
      - Ideal for generating images, but not suitable for fine-tuning
    - `Full` models use all the weights, including the EMA and the non-EMA
      - These models are larger and slower to process, requiring more VRAM to run
      - Ideal for training new models, but not suitable for generating images
    - `Pruned` models have had unnecessary/irrelevant weights removed
      - These models have fewer weights to process, making them faster to run
      - **Note**: Removing weights can sometimes negatively impact the model's accuracy for certain prompts, especially for terms not well-represented in the training data
  - LoRAs (Low-Rank Adaptation)
    - LoRAs are techniques used to adapt a model to specific characteristics or datasets
    - Some models include this within the checkpoint, enhancing the quality of generated images but potentially decreasing compatibility with other LoRAs that could be used in conjunction with the model
  - VAEs (Variational autoencoders)
    - Some models require a VAE to be used in conjunction with them to properly generate images
      - The VAE encodes images into a latent space that the model uses during training
      - At generation time, the model decodes points from the latent space back into images
      - Without the matching VAE, the model can't properly reconstruct the images
    - If a checkpoint specifies a certain VAE requirement, you must use that VAE to achieve proper image generation; otherwise, the results will be suboptimal
  - Model categories
    - `Base` models are the most common and versatile, and are the best choice for use as a base for fine-tuning
    - `XL` models are high-resolution models that can generate more detailed images but are slower to process and require more VRAM
    - `Anime` models specialize in generating anime-style images
    - `Cartoon` models specialize in generating cartoon-style images
    - `Art` models specialize in generating digital art, drawings, paintings, and other artistic styles
    - `Photorealistic` models specialize in generating images that resemble real photographs
    - `Portrait` models specialize in generating images of people
    - `Hybrid` models combine two or more of the previous categories and are the best choice for tasks requiring a mix of styles or variety in the generated images
    - [Model Classification](https://civitai.com/articles/1939/an-attempt-at-classification-of-models-sd-v15) article

## Next Steps

- Using Stable Diffusion WebUI
- Loading image generation models
- Testing different models and checkpoints
- Optimizing configurations and parameters for running models on your device
- Exploring prompting techniques for image generation
- Utilizing Extensions in Stable Diffusion WebUI
- Implementing ControlNet to guide the diffusion process
