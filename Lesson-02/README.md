# Lesson 02: Introduction to Machine Learning, Transformers, and LLMs

In this lesson, we will introduce the basic concepts of machine learning, transformers, and Large Language Models (LLMs) to better understand how Generative AI works, its capabilities, and limitations.

We will gain practical insights into how Machine Learning models function and what they can accomplish by using the Hugging Face Transformers library to run pre-trained models for various AI tasks. This library abstracts away most of the complexities of using Machine Learning and other AI techniques, making it simple and quick to apply these models to real-world problems.

For this lesson, you should have a basic understanding of how to use [Google Colab](https://colab.research.google.com) notebooks. You should also have at least a very basic understanding of the [Python](https://docs.python.org/3.14/) programming language syntax.

## Prerequisites

- Learn the basics of Python programming language syntax
  - [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
- Create an account at [Hugging Face](https://huggingface.co/)
- Create an account at [Google Colab](https://colab.research.google.com)

## Review of Lesson 01

- What is AI
- Capabilities and limitations
- Practical uses of AI
- Using AI for text-to-text tasks
- Does ChatGPT sound like a person?
- Is ChatGPT "intelligent"?

## Introduction to Machine Learning

- How can a computer "learn"?
- [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a broad terminology for a set of algorithms that can learn from and/or make predictions on data
- There are many forms of Machine Learning:
  - **[Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)**: The most common form of machine learning, which consists of learning a function that maps an input to an output based on example input-output pairs
    - Requires a **training dataset** with input-output pairs
    - The algorithm learns from the dataset and can make/extrapolate predictions on new data
  - **[Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)**: A type of machine learning that looks for previously undetected patterns in a dataset with no pre-existing labels
    - Requires a **training dataset** with input data only
    - The algorithm learns from the dataset and can make predictions on new data
  - **[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**: A type of machine learning that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences
    - Requires a **training dataset** with input data and feedback
    - The algorithm learns from the dataset and can make predictions on new data
  - Other models and techniques that can be applied/extended:
    - Semi-supervised learning
    - Self-supervised learning
    - Multi-task learning
    - Transfer learning
    - Meta learning
    - Online learning
    - Active learning
    - Ensemble learning
    - Bayesian learning
    - Inductive learning
    - Instance-based learning
    - And many others

These models have been evolving and improving over the years, aiming to output some form of "intelligence" from the data, mimicking human-like behavior.

- For example, some advanced Machine Learning algorithms use [Neural Networks](https://en.wikipedia.org/wiki/Neural_network) to compute complex functions and make predictions on data in ways that a "normal" program would take billions or more lines of code to accomplish.

This "brain-like" computational approach has been used to extend the capabilities of AI exponentially, far beyond what traditional computing could achieve.

- An example of this is [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), a subset of machine learning that uses neural networks with many [layers](https://en.wikipedia.org/wiki/Artificial_neural_network#Deep_neural_networks) to learn from data and make much more complex predictions.
- Neural Networks like [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) have been used to power many AI applications for tasks such as image and text recognition and generation, computer vision, and many others.
- Currently (as of mid 2024), the most advanced form of Deep Learning is the [Transformers](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) architecture, which has been used to power many AI applications, including the GPT models.
  - Unlike traditional neural networks, transformers can process data in parallel, making them much faster and more efficient.
  - This technical advancement, aligned with favorable market/investment conditions in recent years, has made the current Generative AI boom possible.

To better experiment with and understand how transformers work, we will use samples from the [Hugging Face tutorials](https://huggingface.co/docs/transformers/index), which make it simple and straightforward to start using these tools and techniques without needing to understand the deep technical details of the models beforehand.

## Introduction to Transformers

Transformer was first introduced in the [Attention is All You Need](https://dl.acm.org/doi/10.5555/3295222.3295349) paper in 2017 by Vaswani et al.

> > _We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely._
> > _Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train_.

- Transformer models operate on the principle of **next-word prediction**
  - Given a text prompt from the user, the model can _infer_ which is the most probable next word that will follow this input
- Transformers use **self-attention mechanisms** to process entire sequences and capture long-range dependencies
- Transformers need to be **pre-trained** on a large dataset to properly provide accurate predictions
  - This is why we use Generative **Pre-trained** Transformers models for handling AI tasks
- Architecture of a Transformer model:
  - **Embedding**:
    - Text input is divided into **tokens**
    - Tokens can be words or sub-words
    - Tokens are converted into **embeddings**
    - Embeddings are numerical vectors
    - They capture **semantic meaning** of words
  - **Transformer Block**:
    - Processes and transforms input data
    - Each block includes:
      - **Attention Mechanism**:
        - Allows tokens to **communicate**
        - Captures **contextual information**
        - Identifies **relationships** between words
      - **MLP (Multilayer Perceptron) Layer**:
        - A **feed-forward network**
          - Processes information in one direction, from input to output, without loops or feedback connections
        - Operates on each token independently
        - **Routes information** between tokens
        - **Refines** each token's representation
  - **Output Probabilities**:
    - Uses **linear** and **softmax** layers
    - Transforms processed embeddings
    - Generates probabilities
    - Enables **predictions** for next tokens
- Transformers are much more capable of understanding semantic relationships than traditional neural networks
  - Example: [Google's BERT for search](https://blog.google/products/search/search-language-understanding-bert/)
  - Example: [DeepMind's AlphaFold 2 for protein structure prediction](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
  - Example: [Meta's NLLB for machine translation](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/)

## Experimenting with Transformers

Instead of diving into the deep technical details of transformers, we will use frameworks, tools, and libraries that abstract away the complexities of the computational, mathematical, and statistical work.

In fact, we're going to use pre-made models and shortcuts that make it as simple as calling a function to execute tasks over data passed as parameters.

> Note: It is important to explore these concepts in depth later, so you understand exactly what is happening under the hood. For now, to build functional AI applications as quickly as possible, we will focus on the practical aspects of using these abstractions and simplifications.

- Machine Learning frameworks and tools:
  - [TensorFlow](https://www.tensorflow.org/)
  - [PyTorch](https://pytorch.org/)
  - [JAX](https://jax.readthedocs.io/en/latest/index.html)
- Using a library to abstract away complexities:
  - [Transformers](https://github.com/huggingface/transformers)
- Getting started with a simple Python script
- Using `Pipelines`:
  - Downloading models
  - Using sample data
- Using `Tokenizer` and `Model` shortcuts
- Working with sample `datasets`
- Following a tutorial for an NLP pipeline

## Getting Started with Transformers

Hugging Face's Transformers library can abstract most of the complexities of using Machine Learning and other AI techniques, making it simple to apply these models to real-world problems.

The only concepts you need to fully understand when interacting with this library are: the _configuration_ itself, the _model_ you are using, and the required _processor_ for the task you are trying to accomplish.

- Using [Transformers](https://github.com/huggingface/transformers) Library from Hugging Face
- Using the [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) API for running pre-trained tasks

  - Running these tasks requires almost no previous knowledge in AI or Machine Learning or even programming, thanks to Hugging Face's [Philosophy](https://huggingface.co/docs/transformers/main/en/philosophy)

- Practical exercises:
  - Exercise 1: Getting started with [Google Colab](https://colab.research.google.com)
  - Exercise 2: Running a **Sentiment Analysis** model using Hugging Face's Transformers library with an [example notebook](https://colab.research.google.com/drive/1G4nvWf6NtytiEyiIkYxs03nno5ZupIJn?usp=sharing)
    - Create a Hugging Face [Access Token](https://huggingface.co/settings/tokens) for using with Google Colab
    - Add the token to the notebook's environment variables
      - Open the "Secrets" section in the sidebar of the notebook
      - Click on `+ Add new secret`
      - Enter the name `HF_TOKEN` and paste your secret token in the value field
    - Click on `Grant access` to grant the notebook access to your Hugging Face token to download models
      - The token is required when downloading models that require authentication
  - Exercise 3: Getting started with Hugging Face's [Transformers](https://huggingface.co/transformers/) library with an [example notebook](https://colab.research.google.com/github/huggingface/education-toolkit/blob/main/03_getting-started-with-transformers.ipynb#scrollTo=mXAlr2u76bkg)

## Next Steps

- What is a **pre-trained** model?
- What are these **tokens** and what is **encoding**/**decoding**?
- What is a **prompt**?
- What is **inference**?
- Getting to know GPTs
- Using OpenAI APIs for running GPTs for text-to-text tasks
