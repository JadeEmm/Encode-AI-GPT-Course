# Lesson 23: Running Stable Diffusion Models Locally

In this lesson, we will explore AI Image Generation with Stable Diffusion models by running them locally on our devices.

We'll use various models and techniques for text-to-image generation tasks. The Stable Diffusion WebUI tool will be our primary interface, as it handles many complex tasks required to run these models and provides a simple, intuitive interface for interaction.

While using the Stable Diffusion WebUI, we'll delve into the technical details of how stable diffusion models work, including possible configurations and parameters to optimize these models for our hardware. We'll also explore various sampling and denoising algorithms used to generate images, and learn how to use the API for a more programmatic approach.

By the end of this lesson, we'll investigate different types of models and checkpoints available for download from public hubs. We'll also test the limitations and capabilities of some of these models.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Installation of Python tools on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency in using `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Knowledge of using `venv` to create and manage virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Installation of `git` CLI on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency in using `git` commands to clone repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Installation of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) on your device and all the correct dependencies for your hardware

## Review of Lesson 22

- AI Image Generation
- Stable Diffusion overview
- Running Generative AI Models
- Using Diffusion Models with Python
- Setting Up Stable Diffusion WebUI
- Stability Matrix
- Downloading Models and Checkpoints

## Generating Images with Stable Diffusion WebUI

- **Selecting a Model (checkpoint)**

  - The starting model is the "classic" `runwayml/stable-diffusion-v1-5` [Model](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - Specifically, the version used is `v1-5-pruned-emaonly`, ideal for inference (generating images) due to lower VRAM usage
    - The "full" `v1-5-pruned` version is suitable for fine-tuning but uses more VRAM
      - `EMAonly` refers to the use of only the Exponential Moving Average (EMA) of the weights, a technique used to stabilize model training
      - The "full" version includes all weights (EMA and non-EMA), using more VRAM but providing more flexibility for fine-tuning
    - "Pruned" indicates the removal of unnecessary/irrelevant weights, making the model smaller and faster to process with minimal performance/precision loss
  - In summary, use the starting model for image generation, but consider changing it before attempting fine-tuning

- **Passing a prompt**

  - Similar to the `Text Generation WebUI`, you can pass a prompt to the model to generate an image
  - We'll explore this in practice later

- **Understanding CLIP**

  - Every textual prompt must be encoded using Contrastive Language-Image Pre-Training before being used to generate images
  - CLIP is a neural network that learns to associate images and text, enabling it to "understand" image content and text meaning
  - Most base SD models have a 75 (or 77) token limit for input
    - Prompts exceeding this limit will be truncated
    - Stable Diffusion WebUI allows for "infinite" prompt length by [breaking the prompt into chunks](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#infinite-prompt-length)

- **Prompting techniques**
  - Generally, more specific and objective prompts yield "better" results
  - As with text-to-text models, prompts are only effective when their contents relate to the model's training data
  - Many models allow for configuring specific [attention/emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis) of each term in the prompt
  - These text-to-image models may also follow instructions to avoid content specified in [Negative Prompts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt)

### Configurations

- **Hardware configurations**

  - Edit the `webui-user.sh` (Linux/MacOS) or `webui-user.bat` (Windows) file at the `COMMANDLINE_ARGS` line
    - Using [Low VRAM Mode](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#4gb-videocard-support)
    - Using [CPU only](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-And-Settings#running-on-cpu)

- Configuring the image generation options
  - **Width and height**: The dimensions of the output image
    - Example: Set width to 512 and height to 384 for a 4:3 aspect ratio, or width to 512 and height to 768 for a 2:3 portrait aspect ratio
    - **Important**: Due to the nature of Stable Diffusion models, even a single pixel change in resolution can significantly alter the generated image
      - For different resolutions:
        1. Generate images at the model's native resolution (typically 512 pixels)
        2. Use the high-resolution fix extension to enhance quality for larger sizes
      - Alternative: Manipulate the [Image Generation Seed](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#sampling-method-selection) to maintain similar outputs across resolutions
  - **Batch size**: Number of images generated simultaneously
    - Consumes more VRAM but offers practical advantages:
      - Generates multiple variations at once
      - Allows for efficient selection of "satisfactory" results
      - Reduces the need for repeated single-image generation
- Configuring the sampling options
  - **Sampling steps**: The fixed number of iterations to generate the image
    - Generally, more steps lead to more detailed images but increase generation time
    - **Note**: After a certain point, additional steps may not noticeably improve image quality and could potentially degrade it in some algorithms
  - **Sampling method**: The algorithm used to sample the image at each step
    - Types of sampling methods:
      - Old-School ODE solvers:
        - Numerical methods for solving Ordinary Differential Equations (ODEs)
          - ODEs: Mathematical equations describing the rate of change of a quantity with respect to another
        - Examples:
          - `Euler`: Simplest solver, approximates ODE solution using finite differences
            - Finite differences: Approximate derivatives using discrete intervals
            - While simple, it can be inaccurate for complex systems
          - `Heun`: More accurate but slower version of Euler, uses second-order Runge-Kutta method
            - Runge-Kutta methods: A family of iterative methods for approximating solutions of ODEs
            - Second-order: Uses two evaluations per step, improving accuracy over Euler
          - `LMS` (Linear multi-step method): Uses multiple previous points to calculate the next solution point
            - Improves accuracy by considering the solution's history
            - Can be more stable for certain types of ODEs
      - Ancestral samplers:
        - Stochastic samplers that add noise at each step, introducing randomness
          - Stochastic: Involving random probability distribution or pattern
            - In this context, it means the sampling process isn't deterministic
            - Each run can produce slightly different results, even with the same inputs
        - Examples: `Euler a`, `DPM2 a`, `DPM++ 2S a`, `DPM++ 2S a Karras`
        - **Characteristics**:
          - Faster processing
            - The added randomness can help the sampler explore the solution space more efficiently
          - May produce noisier results
            - The stochastic nature can introduce artifacts or inconsistencies in the output
          - Outputs can vary significantly between runs
            - This variability can be both a strength (for generating diverse outputs) and a weakness (for reproducibility)
      - `Karras` noise schedule:
        - Developed by Tero Karras (creator of StyleGAN)
        - **Key features**:
          - Carefully controls noise input to improve image variety and realism
            - Noise is added in a structured way to maintain image coherence while introducing variation
          - Initializes and learns noise scaling factors during training
            - This allows the model to adapt the noise levels to different parts of the image generation process
          - Generally slower but produces more consistent, less noisy results
            - The careful noise control requires more computation but often yields higher quality outputs
      - `DDIM` (Denoising Diffusion Implicit Models):
        - Variant of denoising diffusion probabilistic models
        - Uses a non-Markovian reverse process
          - Non-Markovian: Future states depend on more than just the current state
            - This allows the model to consider longer-term dependencies in the generation process
            - Can lead to more coherent and globally consistent images
        - **Advantages**:
          - Generates high-quality samples in fewer steps
            - The non-Markovian nature allows for more efficient sampling
          - Relatively simple and easy to train
            - Despite its sophistication, the model architecture is straightforward to implement and optimize
      - `PLMS` (Pseudo Linear Multi-Step):
        - Variant of the linear multi-step method
        - Uses a probabilistic approach, potentially faster than original LMS
      - Diffusion Probabilistic Models:
        - Generate new data samples through iterative denoising
        - Examples:
          - `DPM`: Basic version using a Markov chain
            - Markov chain: A sequence of possible events where the probability of each depends only on the state in the previous event
              - In DPM, each denoising step only depends on the immediately previous state
              - Simple but can be limited in capturing long-range dependencies
          - `DPM-2`: Improved version using a non-Markovian process
            - Allows for consideration of multiple previous states
            - Can capture more complex relationships in the data
          - `DPM++`: Further improvement with new parameterization
      - `UniPC` (Unified Predictor-Corrector Framework):
        - Training-free framework for fast sampling of diffusion models
        - **Key features**:
          - Model-agnostic design: Can work with various types of models
            - Not tied to a specific architecture, making it versatile across different diffusion models
          - Supports various DPM types and sampling conditions
            - Can adapt to different diffusion processes and initial conditions
          - Faster convergence due to increased accuracy order
  - Comparing sampling methods
    - Consider both performance and quality when selecting a method
    - Refer to comprehensive comparisons:
      - [Performance comparison](https://stable-diffusion-art.com/samplers/)
      - [Quality comparison](https://learn.rundiffusion.com/sampling-methods/)
- Classifier Free Guidance (`CFG`) scale
  - This parameter controls how closely the model adheres to your text prompt during the sampling steps
  - Lower values (close to 1) allow for more creative freedom, while higher values (above 10) are more restrictive and may affect image quality
  - **Caution**: Extremely low CFG values may cause Stable Diffusion to "ignore" your prompt, while excessively high values can lead to oversaturated colors
    - [CFG Scale Comparison Article](https://www.artstation.com/blogs/kaddoura/pBPo/stable-diffusion-samplers)
- Seed
  - The initial value used to generate the random tensor in the latent space
  - This value enables the reproduction of identical or highly similar images from the same prompt and settings
  - Variational seeds can yield intriguing results, simulating exploration of the latent space between two defined seeds
    - This technique can create a "blending" effect between two images, with intermediate generations appearing as a gradual transformation
- Built-in extensions
  - **Restore faces**: Scripts and tools to replace distorted faces with more realistic ones
  - **Tiling**: A technique to modify image borders for seamless repetition in a grid
  - **High resolution fix**: Methods and tools for image upscaling
    - **Note**: Generating high-resolution images is challenging for most models, often resulting in lower quality compared to native resolutions
    - **Best practice**: Generate images at the model's native resolution (typically 512 pixels), then use the high resolution fix to enhance quality
- Additional functionalities

  - The Stable Diffusion WebUI offers various image generation tasks beyond text-to-image conversion:
    - **Image to Image**: Generate new images based on existing ones
    - **Sketch**: Create images from rough drawings or annotated images
    - **Inpainting**: Generate images to replace specific areas within larger images

- **CLIP (Contrastive Language-Image Pretraining)**: Tool for extracting potential prompts describing an image

## Prompting Techniques for Image Generation

- **Overall considerations**

  - Optimal techniques for each model type
  - Positive prompts
  - Negative prompts
  - Presence or absence of embeddings in the training data

- Positive prompt guidelines

  - Structure: Clearly specify the major defining elements in the image

    - Subject: "Generate an image of {object}" where {object} is the primary element you want to create
      - Be as specific as possible, considering the representation of the term in the training data
      - Sometimes using a more general term is preferable to specific ones
      - If desired, add an action: "Generate an image of {object} {action}", e.g., "standing", "running", "flying"
    - Specify form, quantity, and adjectives: "Generate an image of {quantity} {quality} {adjective} {object}"
      - Note: Not all models are trained to handle these types of prompts effectively
        - Prompts like "two big sweet red apples" might yield unexpected results in many models

  - Context: Specify image composition elements, such as settings, style, and artistic form

    - Indicate if it's a portrait, landscape, still life, etc.
    - Specify if it should resemble a photograph, painting, drawing, sculpture, etc.
    - Define composition details like background, lighting, colors, etc.
      - Add specifications like "close-up", "far away", "profile", etc.
    - Depending on the model's training, specify the image style, e.g., "impressionist", "surreal", "realistic"
      - If a model is trained on a specific artist's work, you can mention the artist's name
      - Caution: If the model's training data lacks references to a style, results may differ significantly from expectations

  - Refinements: Some models can handle more specific instructions to add nuance to the image

    - Specify the mood, e.g., "happy", "sad", "scary"
    - Indicate time of day, weather, season, etc.
    - Control lighting, color scheme, temperature, detail level, realism level, etc.

  - Tip: To test if a model "understands" a term, try generating an image with just that word as the prompt
    - Example: Before generating "A happy futuristic landscape picture of Darth Vader riding a blue unicorn", try generating "Darth Vader" and "Unicorn" separately to assess their representation in the outputs

- Negative prompts

  - Some models accept negative prompts to instruct certain elements or aspects to be avoided in the generated image
  - Applying negative prompts may reduce the likelihood of certain flaws or deformations but can also limit the overall quality of the generated images
  - Different subjects may require different sets of negative prompts for optimal results
    - For living creatures, consider including: deformed, ugly, too many fingers, too many limbs
    - For objects, use more general terms like: duplicated, blurry, pixelated, low quality, out of frame, cropped
    - For faces, depending on the pose and close-up, consider: poorly rendered face, deformed, disfigured, long neck, ugly
    - For landscapes, include: blurry, pixelated, out of frame, cropped, text, watermark, low quality, poorly drawn

- Adjusting [Attention/Emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)

  - Select specific words in your prompt to emphasize or de-emphasize by surrounding them with special characters
    - The model's tokenizer interprets these special characters as attention markers, adjusting word probabilities accordingly
    - Each model may use different special characters, so consult the documentation for the specific model you're using
    - SD models used by the WebUI employ two types of special characters: `()` and `[]`
      - `(word)` increases attention to the word by a factor of 1.1
      - `((word))` increases attention by a factor of 1.21 (1.1 \* 1.1)
      - `[word]` decreases attention by a factor of 1.1
      - `(word:1.5)` increases attention by a factor of 1.5
      - `(word:0.25)` decreases attention by a factor of 4 (1 / 0.25)
      - To use these characters literally in a prompt, escape them with a backslash (`\`), e.g., `\(word\)` or `\[word\]`
    - Other models may use different special characters like `{}`, `++`, `--`, etc. Consult the documentation for your specific model/tooling

- Prompt examples

  - [Prompt Templates](https://github.com/Dalabad/stable-diffusion-prompt-templates) repository
  - [Prompt Presets](https://openart.ai/presets) from OpenArt

- The [StyleSelectorXL](https://github.com/ahgsql/StyleSelectorXL) extension

## Using Stable Diffusion WebUI API

- Practical exercises
  - Exercise 1: [Generate a simple image](./exercises/00-Generate-Image.md) from text input with default settings
  - Exercise 2: [Generate a landscape picture](./exercises/01-Generate-Landscape.md) with specific prompt and settings
  - Exercise 3: [Generate a portrait picture](./exercises/02-Generate-Portrait.md) and inpaint the generated face
  - Exercise 4: [Generate a picture from a sketch](./exercises/03-Generate-Sketch.md)
  - Exercise 5: [Edit a picture using sketch](./exercises/04-Edit-Sketch.md)

## Useful Configurations for Stable Diffusion WebUI

- Using the [Local API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)
- Installing and managing extensions
- Application configurations
  - Running in [Low VRAM](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Troubleshooting#low-vram-video-cards)
  - Running in [CPU mode](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings#running-on-cpu)

## Generating Images with the API

- Practical exercises
  - Exercise 6: [Enable the API](./exercises/05-Enable-API.md)
  - Exercise 7: [Generate Images](./exercises/06-Generate-Image-API.md) using the API
    - Code [CallAPI.py](./examples/CallAPI.py) and run it

> Note: The API implemented in Stable Diffusion WebUI uses different endpoints from the OpenAI API. You may need to adjust or re-implement API calls in your applications by changing URLs and endpoint names accordingly.

## Generating Images with Python

- Practical exercises
  - Exercise 8: [Photorealistic image generation](./exercises/07-Generate-Photo.md) by running a script with a preconfigured prompt
    - Code [GeneratePhoto.py](./examples/GeneratePhoto.py) and run it
  - Exercise 9: [Cartoon sequence generation](./exercises/08-Generate-Cartoon.md)
    - Code [GenerateCartoon.py](./examples/GenerateCartoon.py) and run it to generate four images of a cartoonish bomb explosion sequence

## Using a ControlNet

- The challenge of controlling the diffusion process
- Using a ControlNet to guide the diffusion process
  - Image consistency
  - Poses and positions
  - Style transfer
  - Colorization
  - Shapes, edges, and outlines
  - Depth maps
  - Semantic segmentation
  - Shuffles
  - Image manipulation with instructions
  - Inpaint masks
  - Tiles
- Installing and using the [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) extension
  - Downloading a [Compatible Model](https://github.com/Mikubill/sd-webui-controlnet?tab=readme-ov-file#download-models)
- Parameters
  - Input image
  - Enable
  - Low VRAM
  - Control Type
  - Preprocessor
  - Model
  - Control Weight
  - Starting Control Step
  - Ending Control Step
  - Control Mode
  - Resize Mode

## Generating Images with a ControlNet

- Practical exercise
- Practical exercises
  - Exercise 10: Experiment with [Human Pose Generation](./exercises/09-Generate-Pose.md) using the ControlNet to control the image generation process

## Next Steps

- Experimenting with more ControlNet features
- Fine-tuning stable diffusion models
- Computer Vision extension
