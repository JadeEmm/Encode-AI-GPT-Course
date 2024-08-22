# Lesson 13: Deep Dive on GPTs

In this lesson, we'll explore the technical intricacies of loading GPTs and the inference process. We'll experiment with various parameters that can enhance model performance and compatibility across different hardware configurations, and evaluate how different models behave under these conditions.

We will investigate and compare various model types, focusing on their key differences in precision, performance, and hardware requirements.

We'll delve deeper into the GPT training process, examining how models can be enhanced with datasets and Low-Rank Adaptations (LoRAs) to improve performance on specific tasks.

To solidify our understanding of GPT functionality, we'll experiment with fine-tuning a simple GPT-2 model to generate text based on information from a local dataset.

## Prerequisites

- Proficiency in using the shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
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
- Installation and execution of [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) on your device
- Creation of an account at [Google Colab](https://colab.research.google.com)

## Review of Lesson 12

- Weekend project
- Interacting with local models
- Replacing OpenAI with local API services in applications
- Testing prompts to improve the quality and precision of generation
- Building an AI application

## Running GPT Models

- LLM files are digital representations of trained neural networks
- These files store:
  - The model's architecture (layers, attention mechanisms, structural components)
  - Learned weights and biases from training
- Model files encapsulate the LLM's knowledge and capabilities
- They enable the model to generate text, perform tasks, and make predictions

- Model files are similar to `.zip` files in that they package large amounts of information compactly
- The `.zip` files are general-purpose containers for various compressed data types
- However, model files are specialized for storing machine learning model parameters and structure
  - Unlike `.zip` files, these model files don't compress or bundle different file types
- Model files encapsulate the trained state of a model for use by machine learning frameworks
- They are designed to be read and interpreted by software that can utilize the neural network for inference

- Model files are not executable by themselves
- They contain all necessary information to run the model
- To use a model file:
  - It must be loaded into a machine learning framework or library
  - The framework interprets the model's architecture and parameters
- This process is called model loading
- Model loading is crucial for deploying trained models for inference or further training

- Text Generation WebUI supports various model loaders
- These loaders can load different types of pre-trained GPT models
- Each model type has its own characteristics and requirements
- We'll explore common model loaders and their use with different hardware configurations

- Model loaders
  - Hugging Face's [Transformers](https://github.com/huggingface/transformers) Library
    - Loads full precision (16-bit or 32-bit) models
      - Models that typically have a clean name without terms like `GGUF`, `EXL2`, `GPTQ`, or `AWQ`
      - Usually, the model files are named `pytorch_model.bin` or `model.safetensors`
    - Full precision models consume significant VRAM, so it's advisable to select the `load_in_4bit` and `use_double_quant` options to load the model in 4-bit precision using `bitsandbytes` if supported on your system
    - This loader can also load GPTQ models and train LoRAs with them, if configured correctly
  - [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) Library using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [CTransformers](https://github.com/marella/ctransformers)
    - Loads `GGUF` models
  - [ExLlamaV2](https://github.com/turboderp/exllamav2)
    - Loads `GPTQ` and `EXL2` models
  - [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) and [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
    - Loads `GPTQ` models
  - [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
    - Loads `AWQ` models
- Model types
  - Full precision
    - These models use the original 32-bit floating-point precision, offering the highest accuracy but requiring more memory and computational resources
    - These models are ideal for training and fine-tuning
  - GPTQ
    - GPT Quantized models use fewer bit widths (e.g., 8-bit or 4-bit) to represent the weights, reducing memory usage and computational requirements at the cost of some accuracy
    - These models are ideal for inference in low-resource environments, such as consumer-grade GPUs or even CPUs only
  - GGUF (and GGML)
    - GGML (GPT-Generated Model Language) is a tensor library designed for machine learning that uses `.ggml` files to store models
    - It was succeeded in August 2023 by the GGUF (GPT-Generated Unified Format) library, which uses `.gguf` files to store models
    - GGUF files can store both full precision and quantized models
    - A GGUF file contains all necessary metadata within the model file, eliminating the need for additional files such as `tokenizer_config.json` that some other models require
  - EXL2
    - EXL2 (ExLlamaV2) is a model format with optimized compression
    - It is based on the same optimization method as GPTQ, but it allows for mixing quantization levels within a model to achieve any average bitrate between 2 and 8 bits per weight
    - These models are optimized for lower latency and can utilize more of the available computational resources to accelerate inference and training processes
  - AWQ
    - AWQ (Activation-aware Weight Quantization) is a quantization technique that adaptively quantizes the model weights based on their importance, striking a balance between accuracy and efficiency
    - Similar to the EXL2 format, AWQ models can yield great performance if given sufficient computational resources
- Configuration tweaks
  - GPUs
    - Many models are optimized for utilizing GPU clusters for running inference and training
    - GPUs are particularly effective for processing GPTs because they can handle the large matrix multiplications required by the models much more efficiently than CPUs
    - The amount of VRAM available on the GPU is a major factor in determining the size of the model that can be loaded and the amount of data that can be processed simultaneously
  - Using consumer-grade GPUs
    - Consumer-grade GPUs are typically less powerful than professional-grade GPUs, but they can still be used to run GPT models
    - GPTQ models are especially well-suited for running on consumer-grade GPUs because they require less memory and computational resources
  - Configuring models to run on consumer-grade GPUs
    - Some loaders like `Transformers` and `GPTQ-for-LLaMa` may allow you to configure the model to offload some of the computation to the CPU, reducing the amount of VRAM required on the GPU, but also significantly slowing down the performance of the models
    - The `llama.cpp` loader uses CPU only by default, so to utilize your GPU, you need to configure the loader to use the GPU by setting the `n-gpu-layers` parameter to a value greater than 0
    - Using quantization can also help reduce the amount of VRAM required to run the model on a consumer-grade GPU
  - CPUs
    - If you don't have access to a GPU, you can still run GPT models on a CPU, preferably using `GGUF` models loaded with `llama.cpp` or `CTransformers`
    - Quantized models like `GPTQ` and `GGUF` with low bit widths are especially well-suited due to lower memory and computational requirements
      - When using `GPTQ`, you can also choose a model that was quantized with a higher `groupsize` parameter, like `1024` (the default, more lightweight), `128` (more balanced), or `32` or less (more accurate but also heavier)
- Optimizing GPT models for your hardware
  - Quantization
    - A technique that reduces the precision of model weights, decreasing memory and computational requirements
    - Trade-off: Quantized models may be less accurate than full precision models
    - Experiment with different bit widths to find the optimal balance between accuracy and efficiency
  - CPU Offloading
    - Some loaders allow offloading computation to the CPU, reducing GPU VRAM usage
    - Useful for consumer-grade GPUs with limited VRAM
    - Trade-off: Significantly slower model performance
  - Disk offloading
    - Certain loaders support offloading computation to disk, minimizing GPU and CPU memory usage
    - Beneficial for systems with limited VRAM or no GPU
    - Trade-off: Substantially slower than CPU offloading; use only as a last resort
  - GPU VRAM usage limits
    - Some loaders allow setting a cap on GPU VRAM usage
    - Helpful for consumer-grade GPUs with limited VRAM
    - When the model exceeds the VRAM limit:
      - It may automatically offload to CPU if configured
      - Otherwise, it will crash with an out-of-memory error
  - CPU RAM usage limits
    - Possible to set a limit on CPU RAM usage
    - If both GPU and CPU limits are set, the model will crash if either is exceeded
      - Some models support disk caching, but this feature is often unreliable and error-prone
  - Disk cache utilization
    - As a final option, use disk cache to store intermediate results and reduce memory requirements
    - Not supported by all loaders; can be slow and prone to errors
  - Context extension
    - Enables running models with larger context sizes by processing input in smaller chunks
  - Workload distribution across multiple GPUs
    - Allows splitting tasks between multiple GPUs for faster processing
    - This is highly dependent on the loader and the model, and may require some manual configuration to work correctly on your system
    - Running GPTs in large clusters is typically done differently than this "artisanal" approach we're using in this bootcamp, with proper load balancing, distribution of the workload, and better orchestration of the GPUs by the model runner
  - Handling sequence lengths
    - Some models have a fixed sequence length that they can process and will crash if the input is longer than this length
    - Some loaders allow for limiting the sequence length to avoid this issue
    - You can experiment with different sequence lengths to find the right balance between accuracy and efficiency for your use case
  - Handling layers
    - Many loaders allow you to configure how the model's layers are handled during inference, providing a trade-off between accuracy and efficiency when using GPUs and CPUs to process outputs
    - If you have a GPU available in your system, you can experiment with loading all the layers on the GPU to speed up the processing and achieve better quality outputs
      - If you encounter an "Out of Memory" error, you can try to gradually increase the number of layers loaded on the CPU until the model loads successfully
  - Using batches
    - Some loaders allow you to configure the batch size of the model, which can greatly affect the performance of the model
    - Larger batch sizes can speed up the processing of the model but may require more memory and computational resources
    - Smaller batch sizes can reduce the memory and computational requirements of the model but may slow down the processing
    - If you're experiencing "Out of Memory" errors, try reducing the batch size to the minimum (one sequence at a time) and see if the model runs

## Overview of GPT Training

- Before pretraining:

  - Initially, a Large Language Model (LLM) is essentially a blueprint with an initialized set of parameters, typically randomly assigned
  - It lacks any "understanding" or knowledge of language and cannot generate meaningful text
  - At this stage, it is merely a chosen architecture (typically based on the Transformer) loaded with a complex mathematical function awaiting data-driven shaping

- Pretraining process:

  - Pretraining exposes this "blank" model to vast amounts of text data
  - Through techniques like next-token prediction or masked language modeling, the model learns to recognize patterns in language, grasp context, and develop a broad knowledge base
  - This process requires immense computational resources and can take weeks or months, depending on model size and available hardware
  - The model's parameters are continuously adjusted to minimize prediction errors, gradually improving its language processing capabilities
  - This phase is computationally intensive, often leveraging specialized hardware like GPUs or TPUs
  - The objective is to enable the model to develop a broad language processing capability without focusing on specific tasks

- After pretraining:

  - Post-pretraining, the model can be fine-tuned on smaller, domain-specific datasets to adapt it to particular tasks or industries, such as summarization or question-answering
  - Fine-tuning further refines the model's parameters, enhancing its effectiveness for specialized applications
  - The model's performance is evaluated using validation sets and various metrics to ensure generalization
  - Finally, the model is deployed, monitored, and periodically updated to maintain performance and relevance
  - Ethical considerations, such as bias mitigation and user privacy, are addressed to ensure responsible usage of the LLM

- Training
  - Architecture
    - GPT's architecture is based on the transformer model, characterized by self-attention mechanisms that weigh the importance of different input data parts to better process the context of words in a sentence
    - The architecture typically includes several layers of transformer blocks, each containing multi-head self-attention and feed-forward neural networks
      - The model needs to be trained to predict the next word in a sentence, given the previous words
    - GPT models are often pre-trained on a large corpus of text data and then fine-tuned for specific tasks
  - Hardware requirements
    - Training GPT models is resource-intensive and requires significant computing power
    - High-performance GPUs or TPUs are generally necessary to expedite the training process
    - As model size increases (with more parameters), hardware requirements scale up, often necessitating distributed training across multiple GPUs or TPUs to accommodate computational load and memory requirements
- Datasets
  - Formats
    - Datasets for training GPT models typically come in text-based formats that the model can easily process
    - Common formats include plain `TXT` files, `CSV` files with text data, or more complex formats like `JSON` or `XML`
    - Some models are trained on specialized formats like TensorFlow's `TFRecord`, which can be more efficient for large datasets
  - Data structure
    - The dataset's structure is crucial for effective training
    - Data should be organized to facilitate the model's learning of relationships between words and sentences
    - This often involves breaking data into manageable chunks, such as sentences or paragraphs
  - Using raw text datasets
    - Raw text datasets can be used directly for training but may require preprocessing to remove unnecessary characters or format the text appropriately
    - The text must be tokenized into a format that the model can process numerically, usually involving splitting the text into words or subwords and converting these into numerical tokens
  - Using JSON objects
    - JSON objects can provide more structured data to the model
    - They may include additional metadata about each piece of text or more complex training examples, such as conversations or question-answer pairs
    - The training pipeline must parse the JSON structure to extract relevant information for the model
  - Considerations about dataset size and variety
    - Dataset size and variety are crucial factors in training a GPT model
    - A larger dataset can help the model learn more about word and sentence relationships, potentially leading to better performance
    - A diverse dataset can improve the model's ability to generalize to new data, enhancing its capacity to generate coherent and relevant text
    - **Larger models** can benefit from bigger datasets, while **smaller models** require more focused, smaller datasets for optimal results
    - Curating the dataset to remove irrelevant, redundant, or low-quality data can enhance model performance
    - For any given prompt, there's an ideal dataset size that will maximize the model's performance for that specific task
      - Adding data beyond this point may decrease the model's performance for that task, while potentially improving its performance for other related tasks
      - Training with an insufficient dataset can lead to **overfitting**, where the model memorizes training data rather than learning underlying patterns
      - Excessive data can result in **underfitting**, where the model fails to learn underlying patterns and struggles with simple tasks directly related to the training data
  - Considerations about dataset quality and data cleaning
    - Dataset quality is crucial for training a GPT model
    - Ideally, every piece of data should be relevant to the model's intended task
    - Some models are more sensitive to data quality than others, potentially requiring more careful dataset curation
      - For example, a model trained on scientific papers may require more meticulous curation than one trained on movie scripts
      - Certain fine-tuning techniques can help the model learn better from data for specific niche tasks
    - Data cleaning is an essential step in dataset preparation, but it can be time-consuming and labor-intensive, often requiring manual inspection and correction
    - Automated tools can assist with data cleaning, but may not catch all errors or inconsistencies
      - In some cases, the cleaning process itself can inadvertently introduce new errors or biases, depending on the parameters used to determine "clean" or "dirty" data
- Training jobs
  - A training job involves updating the model's parameters to minimize the difference between its predictions and expected outputs
  - Iterations
    - An iteration, or epoch, is a single pass through the entire training dataset
    - During each iteration, the model processes data in small batches, updating its weights after each batch to minimize prediction errors
    - This process repeats until the model's performance stops improving significantly
    - The training data is typically divided into smaller batches for computational efficiency
    - Batch size and learning rate (the magnitude of parameter adjustments) are critical factors affecting model performance and training time
  - Convergence
    - Convergence in machine learning refers to the point where the model's performance stops improving significantly
    - It indicates how much the model has learned from the training dataset and whether further training is likely to yield substantial improvements
      - Monitoring convergence is crucial, as training beyond this point can lead to overfitting
    - Convergence is typically monitored by tracking the model's loss or accuracy on a validation set
    - To prevent overfitting, many training processes implement an **Early Stopping Trigger**, halting training when the model's performance on the validation set begins to degrade
  - Measuring and managing loss
    - Loss quantifies the difference between the model's predictions and expected outputs
      - Different loss functions can be used depending on the task and output type
      - Common loss functions include mean squared error (MSE) for regression tasks and cross-entropy loss for classification tasks
    - Lower loss indicates predictions closer to expected outputs, while higher loss suggests greater discrepancies
    - When training on a new dataset significantly different from the pre-training data, it's common to see higher initial loss
      - As the model learns new patterns, the loss should decrease over time
      - However, if the loss decreases too much (below one), the model might start to "forget" pre-training patterns and respond exclusively to the new data
    - Loss is tracked during training to gauge the model's performance cycle by cycle
      - The goal is to reduce loss with each iteration until the model converges and loss stabilizes near the desired value
      - This involves fine-tuning hyperparameters such as the learning rate
      - If the loss remains far from the target but stops decreasing, it may indicate that the model has stopped learning, suggesting a need to adjust training parameters
- Pre-training
  - Pre-training involves training a model on a large dataset to adjust its parameters, enabling it to "understand" underlying patterns in the data, such as relationships between words and sentences in a language or the syntax and semantics of programming languages
  - It involves iteratively training the model on a large dataset, updating its parameters to minimize the error between predictions and expected outputs
- Fine-tuning
  - Fine-tuning involves training a pre-trained model on a new dataset to improve its performance for a specific task
  - It typically uses a smaller dataset than pre-training, adjusting the model's parameters to enhance performance for the new task
  - The model's performance is evaluated using a validation set and various metrics to ensure it generalizes well to new data

## Experimenting with GPT-2

- Introduction to [GPT-2](https://openai.com/research/better-language-models)
  - A groundbreaking language model developed by OpenAI
- Utilizing GPT-2 for training purposes
  - Leveraging the [GPT-2 Simple](https://github.com/minimaxir/gpt-2-simple) package for streamlined implementation
- Key considerations:
  - **Parameter size**: Impact on model complexity and performance
  - **Text dataset**: Importance of quality and relevance
  - **Token limits**: Constraints on input and output length

## Fine-tuning GPT-2

- Practical exercise:
  - Exercise 1: [Fine-tuning GPT-2 Small](./exercises/00-Train-GPT2-Small.md)
    - Objective: Generate text based on a custom dataset
    - Environment: Google Colab instance

## Future Directions

- Exploring Low-Rank Adaptations (LoRAs)
  - Creation and application techniques
- Conducting comparative analyses of various models
- Enhancing GPT capabilities
- Delving into AI Assistants
  - Emphasis on OpenAI's Assistant technology
