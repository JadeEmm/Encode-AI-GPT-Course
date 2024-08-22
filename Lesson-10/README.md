# Lesson 10: Running Models with Text Generation WebUI

In this lesson, we will explore the process of loading models and running inference tasks using Model Loaders. We'll delve into the technical details using the [Text Generation WebUI tool](https://github.com/oobabooga/text-generation-webui).

Text Generation WebUI provides a user-friendly interface for interacting with various [GPT models](https://huggingface.co/models), enabling text generation for numerous tasks and scenarios without sacrificing important configuration and tweaking options. By abstracting complexities and offering a streamlined workflow, this tool empowers us to study the components and capabilities of Generative AI more effectively.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for repository cloning
  - Documentation: [Git](https://git-scm.com/doc)
- Python tools installed on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency with `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Familiarity with `venv` for creating and managing virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) tool installed on your device
  - Follow the repository instructions for installation
  - Ensure you have the necessary dependencies and configurations on your device
- A compatible GPT model downloaded and placed in the correct `models` folder for Text Generation WebUI
- An account at [Hugging Face](https://huggingface.co/)

## Review of Lesson 09

- Python environment setup
- Running AI models on your own device
- Models and hardware requirements
- Model loaders
- Using the GPT4All tool

## Loading Models and Running Inference Tasks

Tools like [GPT4All](https://github.com/nomic-ai/gpt4all) simplify the process of loading models and running inference tasks, even for non-developers. These tools abstract away many configurations, leaving room for basic settings such as CPU thread usage, device selection, and simple sampling options.

For the upcoming lessons, we'll use an open-source developer tool to explore and utilize GPT models on our own devices. While there are more polished and feature-rich tools for end-users, we'll focus on [Oobabooga's Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) due to its extensibility and customizability for developers.

- Cloning [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) repository
- Key Features:
  - Model management, loading, and execution
  - Text-to-text tasks with loaded models
    - Chat, Instructions, and Notebook interfaces
    - Chat templates
  - Model parameter configuration for optimal performance and hardware compatibility
    - Load models with CPU, GPU, VRAM, and RAM limits, quantization, and other helpful configurations
  - Extension management
    - Text-to-speech and speech-to-text integrations
    - Image generation and computer vision
    - Translations
    - Multimodal pipelines and job handling
    - RAG and custom datasets
    - AI Character downloading and management
    - Vector database utilization
    - Custom extension development
  - Local API execution
  - Model fine-tuning
- Installation Preparation
  - Follow the [Official Instructions](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) for most cases
  - Depending on your Operating System and Hardware, you may need to install additional dependencies and configure your environment
  - Note: Not all CPUs and GPUs are compatible with all models and configurations
    - NVIDIA on Linux and Windows currently has the best compatibility (as of March 2024)
    - AMD GPUs on Linux and Windows may work with minor configurations or adjustments
    - Apple and Intel chips may require significant workarounds or may not be compatible
- Installation Steps
  1. Clone the repository
  2. Run the appropriate installation script:
     - `start_linux.sh` for Linux
     - `start_windows.bat` for Windows
     - `start_macos.sh` for MacOS
     - `start_wsl.bat` for Windows Subsystem for Linux
  3. Wait for dependency download and installation to complete
  4. Select your GPU manufacturer and model when prompted
  5. Confirm selection and wait for the entire installation process to finish
  6. Open your browser and navigate to <http://localhost:7860> to test the application
- Alternatives
  - Using a [Docker container](https://github.com/oobabooga/text-generation-webui/wiki/09-%E2%80%90-Docker)
  - Using a [Google Colab notebook](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#google-colab-notebook)

## Downloading Models

Text Generation WebUI allows for downloading and managing models from various sources.

- [Model Download](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#downloading-models) Methods:
  1. Using the `Model` interface:
     - Open the WebUI and navigate to the `Model` tab
     - Click the `Download` button
     - Select the desired model
     - Wait for the download to complete
     - The model will be available in the `Models` tab
  2. Using the command line:
     - Run `python download-model.py <your-model-name>`
     - Replace `<your-model-name>` with the desired model name
       - Example: `python download-model.py "TheBloke/Llama-2-7B-Chat-GGUF"`
- Notable accounts for model exploration and download:
  - [Tom Jobbins](https://huggingface.co/TheBloke)
  - [WizardLM](https://huggingface.co/WizardLM)
  - [EleutherAI](https://huggingface.co/EleutherAI)
  - [Mistral](https://huggingface.co/mistralai)
  - [Stability AI](https://huggingface.co/stabilityai)
  - [OpenAI](https://huggingface.co/openai)
  - [Google](https://huggingface.co/google)
  - [Intel](https://huggingface.co/Intel)
  - [Microsoft](https://huggingface.co/microsoft)
  - [Meta](https://huggingface.co/meta-llama) and [Facebook](https://huggingface.co/facebook)
  - [xAI](https://huggingface.co/xai-org)
  - [ByteDance](https://huggingface.co/ByteDance)
  - [Salesforce](https://huggingface.co/salesforce)
- Model sizes indicating data usage and processing:
  - 7B: Uses 7 billion parameters
  - 13B: Uses 13 billion parameters
  - 30B: Uses 30 billion parameters
  - 70B: Uses 70 billion parameters
    - Examples in model names:
      - `Dr_Samantha-7B-GGUF`
      - `CodeLlama-70B-Python-GPTQ`
- Model types and compatibility considerations:
  - Some models are modified with quantization, pruning, or other techniques for hardware compatibility
  - Models marked with `GGUF`, `GPTQ`, `GGML`, `AWQ`, and similar tags may require specific configurations or tweaking for proper functionality

## Using Text Generation WebUI

We will utilize the Text Generation WebUI tool to run Generative AI tasks on our devices using various models and configurations.

- Practical exercises:
  - Exercise 3: Learn how to use the [Text Generation WebUI](./exercises/00-Text-Generation-WebUI.md)
  - Exercise 4: [Download and run a GPT model](./exercises/01-Running-GPT-Model.md)

## Next Steps

- Utilizing Text Generation WebUI
- Loading and running models
- Exploring Text Generation WebUI extensions
- Setting up and running a local API
