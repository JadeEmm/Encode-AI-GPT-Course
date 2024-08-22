# Lesson 09: Using GPT Models on Your Own Device

In our previous lessons, we have been using APIs to remotely invoke AI models for various tasks. Today, we will delve deeper into how these models are created and how they can be executed on our own devices using a set of tools and libraries that abstract away many of the complexities of the underlying infrastructure.

By the end of this lesson, we will learn that all the AI APIs we've been using from OpenAI and similar companies can be run on our own devices. We'll gain practical knowledge on how to do this for text generation tasks using recommended developer tools.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
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
- [GPT4All](https://github.com/nomic-ai/gpt4all) tool installed on your device
  - Follow the repository instructions to install the tool for your OS

## Review of Lesson 08

- Weekend project review
- Implementation of a multimodal AI chat application
- OpenAI models
- Prompt techniques

## Using Python Tools

- Python
  - [Python](https://www.python.org/downloads/)
- Pip
  - [Pip](https://pip.pypa.io/en/stable/)
- Virtual Environment
  - [Virtual Environment](https://docs.python.org/3/library/venv.html)
- Managing Python versions
  - [pyenv](https://github.com/pyenv/pyenv)
  - [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
- Python packages
- Running [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for text generation

## Running Models Locally

As we've seen previously in this bootcamp, numerous libraries and tools enable us to run GPT models on our own devices.

We've experimented with Hugging Face's **transformers** library, and we could run many models using tools like **pyTorch** and **TensorFlow**. In previous exercises, we utilized the **Google Colab** platform to run the models, allowing us to quickly set up the environment and use the GPUs available in cloud instances provided by that platform.

In this lesson, we will learn how to install the necessary tools on our own devices to run the models locally. This process involves setting up a Python environment, installing the required packages, and configuring the necessary tools to run the models.

It's important to note that depending on your hardware, you might need to install different versions of the packages and tools. Some processors and GPUs are compatible with different versions of the packages and tools, and some may not work at all for certain features we'll be using in the upcoming exercises.

## Using Hugging Face's Transformers

- Setting up the environment
- Installing the necessary packages
- Configuring the required tools
- Running the models
  - Sentiment Analysis
  - Text Generation
  - Text Generation Pipeline
- Hugging Face's Transformers methods and classes

  - [AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
  - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)
  - [BitsAndBytes](https://huggingface.co/docs/bitsandbytes/main/en/index)

- Practical exercises
  - Exercise 1: [Installing Hugging Face's Transformers](./exercises/00-Install-Transformers.md)
  - Exercise 2: [Using a Sentiment Analysis pipeline](./exercises/01-Sentiment-Analysis-Example.md)
    - Code and run [SentimentAnalysisExampleV0.py](./examples/SentimentAnalysisExampleV0.py) and [SentimentAnalysisExampleV1.py](./examples/SentimentAnalysisExampleV1.py) in this exercise
    - More examples on [Transformers quick tour](https://huggingface.co/docs/transformers/quicktour)
  - Exercise 3: [Using a Text Generation pipeline](./exercises/02-Text-Generation.md)
    - Code and run [TextGeneration.py](./examples/TextGeneration.py) in this exercise
  - Exercise 4: [Configuring the Text Generation pipeline](./exercises/03-GPT-Text-Generation-Extended.md) to run a GPT for text generation
    - Code and run [TextGenerationExtended.py](./examples/TextGenerationExtended.py) in this exercise

## Model Loaders Applications

Using Hugging Face's Transformers, we can run many models using tools like `pyTorch` and `TensorFlow`, while configuring the pipelines, models, inputs, and outputs by invoking them inside a Python script. However:

- Configuring these tools and models to work properly within scripts is not always trivial or straightforward
- This process can be overwhelming for beginners
- Numerous other concerns require coding and implementation before we can effectively use the models:
  - Handling server connections
  - Managing model parameters
  - Dealing with caches and storage
  - Fine-tuning
  - Prompt parsing

Several tools can abstract away these concerns and simplify the process for users and developers to use GPT models on their own devices:

- Some of these have binary releases that can be installed and run like any other common software
- Here are a few examples:

1. [GPT4All](https://github.com/nomic-ai/gpt4all): An ecosystem of open-source chatbots and language models that can run locally on consumer-grade hardware.

2. [Ollama](https://github.com/ollama/ollama): A lightweight framework for running, managing, and deploying large language models locally.

3. [Vllm](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference engine for LLMs, optimized for both single-GPU and distributed inference.

4. [H2OGPT](https://github.com/h2oai/h2ogpt): An open-source solution for running and fine-tuning large language models locally or on-premise.

5. [LMStudio](https://lmstudio.ai/): A desktop application for running and fine-tuning language models locally, with a user-friendly interface.

6. [LocalAI](https://github.com/go-skynet/LocalAI): A self-hosted, community-driven solution for running LLMs locally with an API compatible with OpenAI.

7. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui): A gradio web UI for running large language models locally, supporting various models and extensions.

8. [LlamaGPT](https://github.com/getumbrel/llama-gpt): A self-hosted, offline, ChatGPT-like chatbot, powered by Llama 2, that can run on a Raspberry Pi.

These tools offer various features such as model management, optimized inference, fine-tuning capabilities, and user-friendly interfaces, making it easier to work with LLMs locally.

## Testing GPT4All

- Installing GPT4All
- Running the models
- Exercises
  - Exercise 5: [Installing GPT4All](./exercises/04-Install-GPT4All.md)
  - Exercise 6: [Quick Text Generation with GPT4All](./exercises/05-GPT4All-Quick-Generation.md)
    - Download a model and generate a short text response using GPT4All's graphical user interface
  - Exercise 7: [Using a Remote Provider](./exercises/06-GPT4All-Remote-Provider.md)
    - Connect to a remote server with an API Key

## Loading Models with LLaMA C++

- The [LLaMA C++](https://github.com/ggerganov/llama.cpp) tool allows us to load models on our own devices and run them using a C++ interface
- The `llama.cpp` tool can be used to load models and process inferences without programming the instructions in a Python script
- The GPT4All project uses the `llama.cpp` tool to load and run models behind the scenes
  - Since the application is built for ease of use, we have limited control over the model configuration and inference parameters
- Tweaking the model configuration can be crucial to obtaining the best inference results, especially when running tasks on limited hardware resources
- Using remote providers like OpenAI can be an effective way to achieve optimal inference results, particularly when running tasks on limited hardware resources
  - For learning purposes, we aim to have as much control as possible over the model configuration and inference parameters to "better understand" how the entire process works

## Next Steps

- Model loaders
- Tweaking model configurations
- Text Generation WebUI tool
- Running text generation tasks on your own device
