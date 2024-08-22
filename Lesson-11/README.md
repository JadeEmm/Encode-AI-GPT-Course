# Lesson 11: Leveraging Text Generation WebUI Features

In the previous lesson, we explored how to use the Text Generation WebUI application to run GPT models on our own devices for text generation tasks. Now that we're equipped to run these models, we'll delve deeper into the tool's features and learn how to fine-tune configurations for various models and tasks.

We'll cover the diverse interface options for text generation tasks and explore how to enhance our applications using extensions.

Additionally, we'll utilize the Model Loader tab to gain insights into model functionality and examine how our hardware configuration impacts model performance and output quality.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
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
- Proficiency with `git` commands for repository cloning
  - Documentation: [Git](https://git-scm.com/doc)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) tool installed on your device
  - Follow the repository instructions for installation
  - Ensure you have the necessary dependencies and configurations on your device
- A compatible GPT model downloaded and placed in the correct `models` folder for Text Generation WebUI
- An account at [Hugging Face](https://huggingface.co/)

## Review of Lesson 10

- Introduction to Text Generation WebUI
- Model downloading process
- Model loading techniques
- Executing inference tasks

## Exploring Diverse Models

Today, we'll dive deeper into the tool's numerous features and explore its capabilities in running various models and tasks.

- Downloading models using the [Model Hub](https://huggingface.co/models) from Hugging Face
- Model configuration options
- Model loading mechanisms
- Model selection criteria
- Identifying effective instruction-following models
- Executing diverse tasks

## Engaging in Dialogue with a Local GPT Model

- Practical exercises
  - Exercise 1: [Implementing Chat Text Generation](./exercises/00-Chat-Text-Generation.md)
  - Exercise 2: [Utilizing Instruct Text Generation](./exercises/01-Instruct-Text-Generation.md)
  - Exercise 3: [Employing Chat-Instruct Text Generation](./exercises/02-Chat-Instruct-Text-Generation.md)
  - Exercise 4: [Exploring the Notebook mode](./exercises/03-Notebook-Mode.md)

## Advanced Features

- Leveraging [Extensions](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions#built-in-extensions)
- Implementing the [API flag](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API)
- [Fine-tuning](https://github.com/oobabooga/text-generation-webui/wiki/05-%E2%80%90-Training-Tab) models

## Substituting OpenAI APIs

With our newly acquired tool for running GPT models locally, we can now execute **Text Generation** tasks and serve them through a local API.

By utilizing the built-in OpenAI API Extension from Text Generation WebUI, we can replace calls to OpenAI APIs in our scripts and applications with calls to this local API.

- Practical exercise
  - Exercise 5: [Implementing the Text Generation WebUI API](./exercises/04-Using-API.md) as a substitute for OpenAI APIs in a Python script
    - Code and execute [LocalCompletion.py](./examples/LocalCompletion.py) to observe how the Text Generation WebUI API can replace OpenAI API calls

## Future Directions

- Exploring additional Text Generation WebUI features
- Developing an application using the local API
- Conducting an in-depth analysis of GPT models
- Weekend project implementation
