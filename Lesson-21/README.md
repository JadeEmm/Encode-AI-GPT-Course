# Lesson 21: Image Generation

In the previous lesson, we explored Computer Vision models and their ability to correlate image features with textual concepts. This lesson will delve into how this correlation guides image generation processes and how AI can create novel images from textual descriptions.

We will overview image generation techniques involving multi-modal models, neural network-based models, and transformer-based models, examining their advantages and applications.

To grasp how image generation works, we'll explore some of the most common applications of image generation models and interact with online examples.

By the end of this lesson, we will experiment with image generation tasks using the OpenAI Image Generation APIs.

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
- Proficiency in using `venv` to create and manage virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Installation of `git` CLI on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency in using `git` commands to clone repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Basic understanding of `python` programming language syntax
  - [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
- Active account on [ChatGPT](https://chat.openai.com/)
- Active account on [Bing Image Creator](https://www.bing.com/images/create)
- Active account on [OpenAI Platform](https://platform.openai.com/)
  - To run API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 20

- Weekend project
- Computer Vision models
- Multimodal models using Computer Vision
- AI World Scope definition
- Contrastive Language-Image Pretraining

## Image Generation Models

- Generative AI for images
  - Generating pixels that form a coherent image for human perception
  - If an image is generated from a textual description including concepts `A`, `B` and `C`, a human observer should readily recognize the same `A`, `B` and `C` concepts in the generated image
- Evolution of Image Generation algorithms
  - Early approaches: Simple techniques like [cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton) (1940s) and [fractals](https://en.wikipedia.org/wiki/Fractal) (1975) for pattern generation
  - Procedural generation: Utilized in early [video games](https://en.wikipedia.org/wiki/Procedural_generation#Video_games) (1980s) for landscape and texture creation
  - [Markov Random Fields](https://en.wikipedia.org/wiki/Markov_random_field): Applied to texture synthesis in the 1990s
  - [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis): Employed for face generation and manipulation in the early 2000s
  - Non-parametric sampling: Techniques like [image quilting](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) (2001) for texture synthesis and image analogies
  - [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine): Used for learning image features and generation in the mid-2000s
  - [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661): Introduced in 2014, GANs utilize a generator and discriminator network to create realistic images
  - [Variational Autoencoders (VAEs)](https://arxiv.org/abs/1312.6114): Developed in 2013, VAEs learn to encode and decode images, enabling generation of new samples
  - [Pixel RNNs](https://arxiv.org/abs/1601.06759) and [Pixel CNNs](https://arxiv.org/abs/1606.05328): These models, introduced in 2015-2016, generate images pixel by pixel using recurrent or convolutional neural networks
  - [Deep Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/dbm.pdf): Proposed in 2009, these energy-based models can generate images by learning probability distributions over pixel values
  - Autoregressive models: [PixelCNN++](https://arxiv.org/abs/1701.05517) (2017) improved upon earlier pixel-based models for sequential image generation
- Image Generation with Transformers

  - The [transformer architecture](https://arxiv.org/abs/1706.03762) used in GPTs can also be implemented for image generation models
  - Transformers can be trained to relate visual, textual, or audio concepts, allowing for versatility in many AI models
  - [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929) adapt the transformer architecture specifically for image tasks, including generation
  - **Transformer-based models** like [DALL-E](https://openai.com/dall-e-2) and [Stable Diffusion](https://stability.ai/stable-diffusion) have achieved state-of-the-art results in image generation

### Overview of Image Generation Techniques

- Multi-modal Approach

  - Multi-modal AI applications process and generate data in various formats, including text, image, video, and audio
  - These models create connections between different types of data, enabling more versatile and comprehensive AI systems
  - Benefits of multi-modal approaches:
    - Enhanced contextual associations and relationships between different data types
    - Improved ability to generate more coherent and contextually relevant outputs
    - Potential for more natural and intuitive human-AI interactions
      - A non-technical user can relatively easily prompt for a good generation even without using **prompt engineering** techniques
  - Example: ChatGPT with DALL·E
    - Utilizes unsupervised learning to generate images from text descriptions
    - Employs transformer language models to learn the relationship between text and images from large datasets
    - Demonstrates the ability to create novel, creative images based on complex textual prompts
    - Showcases the potential for AI to bridge the gap between linguistic and visual representations
    - DALL·E is fully integrated with [ChatGPT](https://chat.openai.com/), allowing it to automatically generate tailored, detailed prompts for DALL·E 3 when prompted with an idea
      - Users can request adjustments to generated images with simple instructions

- Neural Network-based Models

  - Neural networks, traditionally used for text processing, have been adapted for image generation tasks
  - These models learn to represent and generate complex visual patterns through contrastive training on large datasets
  - Key architectures used in image generation:
    - **Generative Adversarial Networks (GANs)**:
      - Utilize a generator and discriminator network to create realistic images
      - The generator creates images, while the discriminator attempts to distinguish real from generated images
      - This adversarial process leads to increasingly realistic image generation
    - **Variational Autoencoders (VAEs)**:
      - Learn to encode images into a compressed latent space and then decode them back into images
      - Allow for generation of new samples by manipulating the latent space
      - Useful for tasks like image reconstruction and style transfer
    - **Convolutional Neural Networks (CNNs)**:
      - Particularly effective for image-related tasks due to their ability to capture spatial hierarchies
      - Can be combined with other architectures to improve image generation quality
    - **Recurrent Neural Networks (RNNs)**:
      - Used for sequential image generation tasks
      - Useful for generating images with temporal dependencies, such as video frames

- Transformer-based Models
  - Transformer architecture, originally designed for natural language processing, has been successfully applied to image generation
  - These models treat image generation as a sequence prediction task, similar to language modeling
  - Key features of transformer-based image generation models:
    - Self-attention mechanisms:
      - Allow the model to weigh the importance of different parts of the input data
      - Enable the model to capture global dependencies in the concepts present in the images
    - Ability to capture complex dependencies:
      - Enables generation of high-quality, coherent images
      - Allows for improved representation of long-range relationships in visual data
    - Scalability:
      - Can handle large datasets and long-range dependencies effectively
      - Allows for training on diverse and extensive image collections
    - Parallelization:
      - Allows for efficient computation and training
      - Enables faster generation of high-resolution images
  - Examples of transformer-based image generation models:
    - **DALL·E (OpenAI)**:
      - Generates images from textual descriptions
      - Demonstrates impressive creativity and representation of complex concepts
    - **Stable Diffusion (Stability AI)**:
      - Open-source model capable of generating high-quality images from text prompts
      - Known for its efficiency and ability to run on consumer hardware
    - **Imagen (Google)**:
      - Produces photorealistic images with strong text alignment
      - Showcases advanced capabilities in representing and rendering complex scenes
  - These models have achieved state-of-the-art results in image generation tasks, demonstrating the versatility and power of the transformer architecture
  - Ongoing research focuses on improving the efficiency, controllability, and ethical use of these powerful image generation models

## Overview of Stable Diffusion for Image Generation

- Similarities with GPT Language Models (LLMs)
  - Underlying architecture and operational principles
  - Use of transformer-based generative architecture
- Key differences from GPT models
  - Adapted to handle visual data instead of textual data
  - Processes pixel or feature sequences rather than textual sequences
- Image generation process
  - Initiated with a text prompt or partially completed image
  - Input encoded into latent space representation
  - Transformer iteratively refines the encoded representation
    - Guided by patterns learned during training
  - Refinement continues until a coherent, contextually relevant image is generated
- Core concepts
  - **Latent space representation**: Compressed encoding of input data
  - **Iterative refinement**: Gradual improvement of the image through multiple steps
  - **Contextual relevance**: Alignment with the initial prompt or partial image
  - **Diffusion process**: Gradual denoising of a random noise input to generate an image
  - **Attention mechanisms**: Allowing the model to focus on relevant parts of the input
  - **Conditioning**: Incorporating text prompts or other inputs to guide image generation
- Capabilities
  - Generates images from textual descriptions or incomplete images
  - Demonstrates "understanding" of visual data and context
  - Infers pixels and features related to concepts in the prompt
    - Similar to how GPT infers tokens related to other tokens
- Generation process details
  - Starts from a random noise image
  - Predicts and adds details in each iteration
  - Decodes latent representation into a coherent, detailed image
- Advantages of the approach
  - Produces highly detailed and contextually relevant images
  - Leverages transformer's ability to handle complex dependencies and relationships
  - Applicable to both text-to-image and image-to-image tasks
- Example use cases for stable diffusion
  - Digital art creation: Generating unique artworks from textual descriptions
  - Product design: Rapidly prototyping new designs based on specifications
  - Architectural visualization: Creating realistic renderings from blueprints
  - Fashion design: Generating new clothing designs based on style descriptions
  - Game asset creation: Producing diverse characters and environments
  - Book illustration: Generating images to accompany written stories
  - Advertising: Creating custom visuals for marketing campaigns
  - Educational materials: Illustrating complex concepts for textbooks or courses
  - Film and animation: Generating concept art and storyboards
  - Interior design: Visualizing room layouts based on client preferences
- Ethical and Legal Considerations
  - Copyright and intellectual property:
    - Potential infringement when generating images based on copyrighted works
    - Disputes on ownership of AI-generated images
  - Bias and representation:
    - Risk of perpetuating societal biases present in training data
    - Importance of diverse and inclusive training datasets
  - Misinformation and deepfakes:
    - Potential misuse for creating misleading or false images
    - Need for detection methods to identify AI-generated content
  - Privacy concerns:
    - Risk of generating images that violate individual privacy rights
    - Importance of consent when using personal likenesses
  - Environmental impact:
    - High computational requirements leading to significant energy consumption
    - Need for more efficient algorithms and sustainable computing practices
  - Ethical use guidelines:
    - Importance of developing and adhering to responsible AI principles
    - Need for transparency in disclosing AI-generated content
  - Regulation and governance:
    - Need for clarity on legal frameworks for AI-generated content
    - Balancing innovation with responsible development and use

## Generating Images with Stable Diffusion

- Overview of Stable Diffusion

  - **Stable diffusion**: A type of **generative model** using a **diffusion process** to generate images
    - **Generative model**: Machine learning models that create new data similar to training data
    - **Diffusion**: Process of gradually adding noise to data and learning to reverse it
  - Core concept: Iteratively apply transformations to a **random noise vector** to produce a coherent image
    - **Noise vector**: Randomly initialized values serving as the starting point
    - **Transformations**: Mathematical operations shaping the noise into an image
  - Each step moves the noise vector closer to resembling a realistic image

- Running a Diffusion Model

  - Process: Iteratively apply transformations to a random noise vector
  - Goal: Produce a coherent image
  - Method: Select transformations that guide the noise vector towards resembling a realistic image
  - Termination: Predetermined number of steps or when output meets quality criteria

- How it Works

  - Stable diffusion employs two main steps: **forward diffusion** and **reverse diffusion**

  1. Forward Diffusion (Noising Process)
     - Begins with an image, gradually adds **Gaussian noise** over steps until it becomes pure noise
     - Modeled as a **Markov chain**: Each step depends only on the previous one
     - Utilizes conditional Gaussian distributions
     - Transforms data distribution into a **tractable distribution** (e.g., isotropic Gaussian)
       - **Tractable distribution**: Easy to sample from and compute probabilities
       - **Isotropic Gaussian**: Multivariate normal distribution with independent variables and uniform variance
  2. Reverse Diffusion (Backward Denoising)
     - Reverses the noising process, converting noise back into an image
     - Employs a **neural network** to predict and subtract added noise at each step
     - Network is conditioned on factors such as current noise level and possibly text embeddings
       - **Embedding**: Dense vector representation of discrete data in continuous space
     - Iterative process: Begins from pure noise, gradually reduces noise to reconstruct an image
     - Output: Sample resembling a natural image or corresponding to text description

- Training the Model

  - Optimizes a **loss function**: Measures difference between predicted and actual added noise
  - **Loss function**: Guides optimization of model parameters

- Image Generation
  - Initiates from random Gaussian noise image
  - Iteratively denoises using learned reverse diffusion process

> The best way to understand visual AI generation is with visual examples. The [The Illustrated Stable Diffusion](http://jalammar.github.io/illustrated-stable-diffusion/) article is an excellent resource to understand this process visually.

### Image Generation Examples

- [DALL·E-2 and DALL·E-3](https://labs.openai.com/) integrated with [ChatGPT](https://chat.openai.com/)
- [DALL·E-3](https://openai.com/index/dall-e-3/) integrated with [Designer](https://www.bing.com/images/create) (formerly Bing Image Creator)
- [Midjourney](https://www.midjourney.com/showcase)
- [DreamStudio](https://beta.dreamstudio.ai/generate)
- [Stable Diffusion](https://stablediffusionweb.com/)
- [Canva Text to Image](https://www.canva.com/your-apps/text-to-image)
- [Adobe Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html)
- [Imagen by Google](https://imagen.research.google/)
- [Craiyon](https://www.craiyon.com/) (formerly DALL-E mini)
- [Picsart](https://picsart.com/create/editor)
- [Image-FX](https://aitestkitchen.withgoogle.com/tools/image-fx)

- Practical exercise
  - Exercise 1: [Generate Images with AI Tools](./exercises/00-Generate-Images.md) to compare outputs

## Utilizing OpenAI Image Generation APIs

- OpenAI API endpoints for [Image Generation](https://platform.openai.com/docs/api-reference/images)
- Generating images with [DALL·E](https://platform.openai.com/docs/guides/images) from text prompts
  - DALL·E-2 vs DALL·E-3
    - DALL·E-2: Earlier version with "good" image quality but less coherence with complex prompts
    - DALL·E-3: Latest version with "improved" understanding of prompts, "enhanced" image quality, and more accurate text rendering
  - Quality
    - Options typically include 'standard' and 'hd' (high definition)
    - Higher quality settings produce more detailed images but may require longer generation times
    - This parameter is only supported for DALL·E-3
  - Size
    - Must be one of `256x256`, `512x512`, or `1024x1024` for DALL·E-2
    - Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for DALL·E-3
    - Pricing varies based on size
  - Styles
    - Can be either `vivid` or `natural`
    - Only available for DALL·E-3
- Generating images with other images as input
  - Edit
    - Prompt: Textual description of desired changes to the original image
    - Mask: Specifies which areas of the image to modify, allowing for targeted edits
  - Variations
    - Creates multiple versions of an input image while maintaining its core elements
    - Useful for exploring different artistic interpretations or styles

### Generating Images with DALL·E Using Python

- Practical exercises
  - Exercise 1: [Generate Images with DALL·E](./exercises/01-Generate-Images-DALL-E.md)
    - Code and execute [Image-Generation.py](./examples/Image-Generation.py) to generate an image from a text prompt
  - Exercise 2: [Generate Image Variations with DALL·E](./exercises/02-Generate-Image-Variations.md)
    - Code and execute [Image-Generation-Variations.py](./examples/Image-Generation-Variations.py) to generate variations of an image

## Next Steps

- Hugging Face's Stable Diffusion Pipelines
- Image generation models
- Stable diffusion process
- Running models locally
- Stable Diffusion WebUI
- Environment setup
- Downloading models
