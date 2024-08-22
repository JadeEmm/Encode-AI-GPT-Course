# Lesson 24: AI Image Processing

In this lesson, we'll delve deeper into how ControlNets can be utilized to _intervene_ in the diffusion process, guiding the generation flow to conform the final image to specific structures we define.

We'll then explore how diffusion models can be enhanced and fine-tuned using LoRAs (Low-Rank Adaptations) and VAEs (Variational Autoencoders) to significantly expand their image generation capabilities. We'll experiment with image datasets and techniques to effectively correlate elements in an image with approximated text descriptions.

In the latter half of this lesson, we'll focus on extracting information from images. We'll introduce an extension for Stable Diffusion WebUI for Computer Vision and demonstrate its application in image classification tasks, both within the WebUI and via API.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Installation of Python tools on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Competence in using `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Knowledge of `venv` for creating and managing virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Installation of `git` CLI on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency in using `git` commands to clone repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Installation of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) on your device with all necessary dependencies for your hardware
- Installation of the [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) extension with all required dependencies for your hardware
  - Download at least one [Compatible Model](https://github.com/Mikubill/sd-webui-controlnet?tab=readme-ov-file#download-models) and place it in the `models` folder within the ControlNet extension directory
- Installation of the [Clip Interrogator Extension](https://github.com/pharmapsychotic/clip-interrogator-ext)

## Review of Lesson 23

- Stable Diffusion WebUI
  - Installation
  - Running
  - Configuration
  - Extensions
- Text-to-image tasks
- Models and checkpoints
- Model comparison
- Prompt engineering for image generation
- Configuring Stable Diffusion WebUI
- Capabilities and limitations of stable diffusion models
- ControlNet extension for Stable Diffusion WebUI
- Utilizing ControlNet to guide the diffusion process

## Leveraging ControlNet Models

- Picture controls
  - Colorization/recolor
  - Edge/outline detection by color/brightness/contrast
- Detection capabilities
  - Depth
  - Segmentation
  - Borders
  - Pose
  - Objects
- Implementing custom controls in ControlNet
  - Brightness
  - Lighting
- Compatibility considerations

- Practical exercise
  - Exercise 1: Generate [QR Code Art](./exercises/00-Generate-QRCode.md) linking to a website using ControlNet

## Fine-Tuning Models with Stable Diffusion WebUI

- Datasets
  - Quantity and quality: Larger datasets generally yield superior results, but quality is paramount High-quality, diverse images are crucial for effective training
  - Image resolution: Models are designed to work with images of specific resolutions
- Captions
  - Accurate and descriptive captions for each image in the dataset are essential for training the model to "understand" the relationship between text and images
  - Captions should be detailed and relevant to the specific content of each image
- Starting with a base model
  - `runwayml/stable-diffusion-v1-5` (default): A widely used and versatile base model that serves as an excellent starting point for many fine-tuning tasks
  - Other base models can be employed depending on the specific requirements of your project
- Generating captions with Computer Vision models
  - Automated caption generation using pre-trained vision models can save time and provide consistent descriptions
  - Models like CLIP (Contrastive Language-Image Pre-training) can be utilized to generate relevant captions for large datasets
- Referencing images for styles
  - Using reference images helps guide the model to produce outputs in specific artistic styles or visual aesthetics
  - This approach is particularly useful when fine-tuning for a particular artist's style or a specific visual theme
- The training process

  - Checkpoints: Saving model states at regular intervals allows for resuming training and selecting the best-performing version
  - Samples: Generating sample images during training helps monitor progress and adjust parameters if necessary
  - Steps: The number of training iterations; more steps can lead to "better" results but increase training time
  - Intervals: Frequency of evaluation and checkpoint saving during training
  - Batches: Number of images processed in each training step; larger batches can accelerate training but require more memory
  - Learning Rate: Controls how quickly the model adapts to new information; requires careful tuning for optimal results
  - Image Processing: Techniques like augmentation and normalization to improve model generalization
  - Concepts: The specific ideas or themes the model is being trained to "understand" and generate
  - Filewords: Keywords associated with specific files or images in the dataset
  - Prompts: Text inputs used to guide the model's image generation during training and testing
  - Image Generation: The process of creating new images based on the trained model and input prompts

- Fine-tuning the model for a new concept
  - Using a [Google Colab notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
    - Make a copy of the notebook and run it in your own environment: This allows for customization of the training process and progress saving
    - Use a [Hugging Face Token](https://huggingface.co/settings/tokens) with `write` permissions to save your fine-tuned model and its concepts: This enables storage and sharing of your custom model on the Hugging Face platform

## LoRAs and VAEs

- LoRAs (Low-Rank Adaptations)
  - Definition and purpose
  - How LoRAs work with base models
  - Advantages of using LoRAs
    - Smaller file sizes
    - Faster training times
    - Easier to share and distribute
  - Creating custom LoRAs
  - Combining multiple LoRAs
- VAEs (Variational Autoencoders)
  - Definition and purpose in image generation
  - Functionality with Stable Diffusion models
  - Types of VAEs
    - Standard VAEs
    - Improved VAEs (e.g., kl-f8-anime2)
  - Selecting and implementing different VAEs
  - Impact on image quality and style

## Utilizing Stable Diffusion WebUI for Image Descriptions

- The `PNG Info` tool
- Installing and using the [Clip Interrogator Extension](https://github.com/pharmapsychotic/clip-interrogator-ext)

- Practical exercise:
  - Exploring the [img2text features](./exercises/01-Img2Text-Features.md) of the Stable Diffusion WebUI

## Integrating AI Functionalities

- Wrapping up the bootcamp
- Building AI applications
- Integrating Computer Vision, Text Generation and Image Generation

- Practical exercises:
  - Create a new Next.js application
  - Add a [page](./exercises/02-Computer-Vision-Page.md) to upload images and process them with Computer Vision
  - Add a [page](./exercises/03-Chat-Page.md) to interact with a text generation model
  - Add a [page](./exercises/04-Image-Generation-Page.md) to generate images with an image generation model

## Final Project

1. Ideation
2. Planning
3. Prototyping new tools
4. Sketching the user interface
5. Building the application
6. Testing
7. Deployment
8. Submission and presentation
