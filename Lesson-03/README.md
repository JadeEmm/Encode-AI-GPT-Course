# Lesson 03: Introduction to GPTs

In this lesson, we will delve into the introduction of GPTs (Generative Pre-trained Transformers) and explore their functionality.

We will gain a better understanding of how transformers are used to train and compute outputs from models, and we will evaluate in depth the capabilities and limitations of current GPT applications.

We will practice more with using the Hugging Face Transformers library to run pre-trained models for text-to-text tasks with an LLM, focusing on sampling methods and tweaking model configurations to understand the underlying mathematics and algorithms.

## Prerequisites

- Learn the basics of **Python** programming language syntax
  - [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
- Create an account at [Hugging Face](https://huggingface.co/)
- Create an account at [Google Colab](https://colab.research.google.com)

## Review of Lesson 02

- What is Machine Learning
- What are Transformers
- Practical uses of Transformers
- Using Transformers for text-to-text tasks with Hugging Face's Transformers library
- What are LLMs
- Using pre-trained LLMs for text generation
- Running an LLM inference for text generation with Hugging Face's Transformers library
- What is a _pre-trained_ model?
- What are **tokens** and what is _encoding_/_decoding_?
- What is a _prompt_?
- What is _inference_?

## Running Generative AI tasks with Transformers

- Creating tokens from strings using the [Tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#tokenizer) methods for LLMs
- Generating text by running LLMs with [Generate](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) API
  - [Text generation with LLMs](https://huggingface.co/docs/transformers/main/en/llm_tutorial#generation-with-llms) uses _Autoregressive generation_ to iteratively infer outputs from a given input from a model
  - There are many different [decoding methods for language generation with Transformers](https://github.com/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)

- Practical exercises
  - Exercise 1: Running **NLP** tasks with the [BERT](https://arxiv.org/abs/1810.04805) model using Hugging Face's Transformers library with an [example notebook](https://colab.research.google.com/drive/1PHv-IRLPCtv7oTcIGbsgZHqrB5LPvB7S)
    - Small fix to the encoding section using [this code](https://github.com/curiousily/Getting-Things-Done-with-Pytorch/pull/8/files)
  - Exercise 2: Using [Google Colab](https://colab.research.google.com) to run the [Generative AI Task Summary](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/task_summary.ipynb) notebook example
    - Comparing pipelines to running tasks directly with PyTorch and TensorFlow
    - Introduction to the concept of **Causal Language Modeling**
    - Use **!pip install transformers==4.38.2** to fix the import for **top_k_top_p_filtering** in this notebook

## Introduction to GPTs

- GPTs (Generative Pre-trained Transformers) are a type of LLMs (Large Language Models) trained to generate **outputs** (inference) from **inputs** (prompts) based on their **training data** (pre-training)
- Although GPTs can follow complex instructions, like chatting, answering questions, generating stories, and completing code, they do not possess true intelligence or "understanding" of the **inputs** and **outputs** they process
  - GPTs don't actually "understand" the **inputs** and **outputs** they process; they learn from the training data (using **Machine Learning**) how to generate outputs from inputs based on **patterns** and **probabilities** observed in the training data
  - GPTs don't retain information they process after the training phase; they "forget" the **inputs** and **outputs** after processing
    - GPTs can store the chat history in **context** while processing tasks, but this is only temporary storage for the current chat/inference session
    - When the chat/inference session is deleted, the **context** is lost, and the GPT "forgets" everything it processed
  - GPTs don't "learn" or "improve" each time they process an inference, regardless of how many times they process similar **inputs** and **outputs**
    - However, GPTs can be **fine-tuned** to process specific **inputs** and **outputs** after the initial training phase to perform better on specific tasks and/or datasets

### GPT Applications

Some examples of applications using GPTs:

- [ChatGPT](https://chat.openai.com/)
- [Google Gemini](https://gemini.google.com/) (formerly known as Bard)
- [Bing AI](https://www.bing.com/chat) (also known as Copilot)
- [Claude AI Chat](https://claude.ai/chats)
- [Pi](https://pi.ai/talk)
- [Grok](https://grok.x.ai/)

All of these applications use various techniques and models to process different types of tasks (also known as modalities), but they all share similar limitations and capabilities of GPTs, to varying degrees and in different ways.

There are ongoing discussions about evolving these applications to reach [AGI](https://en.wikipedia.org/wiki/Artificial_general_intelligence) (Artificial General Intelligence) capabilities using GPTs and similar models. As of mid-2024, this remains a complex and long-term goal that is not yet feasible with the current state of the art in AI and Machine Learning (and may never be).

### How GPTs Work

- Model training
  - GPTs are pre-trained models that are trained on large and diverse datasets of texts to compute **patterns** and **probabilities** between the contents present in the training data
    - For example, a model trained on terabytes of chemical and biological texts may be able to generate correlations between chemical component names and illness symptoms with much more accuracy and reliability than a model trained on terabytes of literature and poetry texts
    - Generally, the more data and the more diverse the data, the better the model will be able to generate outputs based on the **patterns** and **probabilities** observed in the training data
    - Current state-of-the-art models are trained on datasets with hundreds of terabytes of texts and other content, and the training process can take weeks or even months to complete even on powerful hardware
  - The process of training these models involves sophisticated **Machine Learning** techniques, as we studied in the previous lesson
- Tokens encoding and decoding
  - All calculations in these models are performed on **tokens** that are _encoded_ and _decoded_ from character strings, rather than on the raw character strings themselves
  - Each word (or fragment of letters) is broken down into tiny pieces ([n-grams](https://en.wikipedia.org/wiki/N-gram)) that are used to compute the **patterns** and **probabilities** between the contents present in the training data
  - The _encoding_ process converts character strings into these **tokens**, and the _decoding_ process converts these **tokens** back into character strings
  - This tokenization process allows the model to handle a wide range of languages and character sets efficiently
- Transformers
  - The process of relating tokens to each other in the training data is done by [Transformer](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) neural networks, which we saw in the previous lesson
  - These transformers use self-attention mechanisms to capture complex relationships between different parts of the input data
  - Self-attention vs Multi-head attention: Self Attention would be like considering how each word relates to every other word to understand the overall meaning. Multi-head self-attention would be like reading the sentence several times, each time focusing on a different aspect like grammar, word relationships, or sentiment.
  - Transformers are used both in the training phase and in the inference phase of these models to calculate the relations between tokens and generate outputs
- Inference
  - The process of generating outputs from inputs in a transformer is called inference, and can be accomplished much more quickly for **prompting** than the training phase, since the **patterns** and **probabilities** between the contents present in the training data are already computed and stored in the model
  - Inference is the process that a trained machine learning model uses to draw conclusions from brand-new data. An AI model capable of making inferences can do so without examples of the desired result. In other words, inference is an AI model in action.
  -     For example: Email security: A machine learning model can be trained to recognize spam emails or business email compromise attacks, then make inferences about incoming email messages, allowing email security filters to block malicious ones
  -     https://www.cloudflare.com/en-gb/learning/ai/inference-vs-training/
  - For every **prompt** provided, the model calculates the most probable **output** based on the model data using various decoding strategies
  - Since the model doesn't "understand" the concepts of _correct_ or _incorrect_, _meaningful_ or _nonsensical_, _helpful_ or _toxic_, it is up to the person or system passing the **prompt** to the model to _judge_ the **output** generated by the model according to their own criteria and standards
- Prompting
  - After the model has been trained, it won't execute anything on its own unless it's _prompted_ to do so
  - The **prompt** being passed to the model may vary greatly in format, depending on the task being executed and the model configurations
    - For example, a **prompt** for a chatbot may be a question like "_What is your name?_", while a **prompt** for a code generator may be a code snippet like "_for i in range(10): print(i)_"
    - Most models have certain _structures_ for prompts, allowing for multiple _actors_ to interact inside the model, usually with a "system" actor for answering things that a "user" actor requests from it
  - The **prompt** is the input that the model will process to generate the output, and it can be as simple as a single word or as complex as a full book
  - The **prompt** is the most important part of the process of generating outputs from inputs, and it can be _engineered_ in many different ways to _guide_ the model to generate more _meaningful_ outputs
- Context
  - GPTs have a limited _context_ maximum length for processing **prompts**, and they can store the chat history in **context** while processing tasks, but this is only temporary storage for the current chat/inference session
  - When passing too much data in a **prompt**, the model may not be able to process it all correctly, and may even disregard some parts of the **prompt** or of the **training data** based on the model configurations and parameters
  - The context size usually ranges from a few hundred to a few thousand tokens, with some exceptions up to one million tokens context size
    - This means that the more information you include in the **prompt**, the less _token space_ the model will have to evaluate the trained data against what is being asked
  - Managing context effectively is crucial for obtaining accurate and relevant responses from the model
- Fine tuning
  - If GPTs trained on random internet text were not adjusted, they would merely regurgitate random internet texts, giving responses that are completely irrelevant, inaccurate, or even nonsensical
  - To save space in the **context** while improving the accuracy of the outputs, the **fine tuning** process can be used to _guide_ the model to generate more _meaningful_ outputs
  - In the **fine tuning** process, the model is trained to prioritize reliable sources and accurate information, reducing the likelihood of generating answers based on misinformation or low-quality internet texts
  - These processes often include incorporating human feedback to correct errors and biases in the model's responses. This iterative process helps in aligning the model's outputs with human values and expectations
  - Another common practice is to apply targeted training, which involves fine-tuning the model on a specific dataset that is relevant to the desired application
    - This targeted approach helps the model learn context and nuances pertinent to particular topics or industries
- RAG
  - The **RAG** (Retrieval-Augmented Generation) process is a technique that combines a language model's generative abilities with an external information retrieval system
  - Instead of passing all the context for a question in the prompt, RAG first identifies relevant information from a large database, then incorporates this data into the language model's query
  - This method significantly improves the model's precision and accuracy, particularly for queries requiring specific information
  - This is more efficient than passing all the information in the prompt, and is also much simpler than fine-tuning the model with new data
  - RAG allows models to access up-to-date information without requiring constant retraining, making it particularly useful for applications that need to provide current and accurate information

### Limitations of GPTs

To better understand the capabilities of GPTs, let's highlight some of the most impactful implications of their limitations:

- Lack of "understanding"
  - Questions like "_Who are you?_" and "_What is your name?_" are not "understood" by GPTs; they simply generate outputs based on patterns and probabilities observed in the training data
  - If the training data contains texts like "_I am a chatbot_" and "_My name is ChatGPT_" frequently associated with these questions, the GPT will generate outputs like "_I am a chatbot_" and "_My name is ChatGPT_" when asked
  - If the training data is some amount of text where these questions are frequently associated with other words and phrases, the GPT will generate outputs based on these associations
    - For example, a GPT trained on the Don Quixote book might generate outputs like "_I am Don Quixote_" and "_My name is Alonso Quijano_" when asked these questions
  - There are techniques like **Fine Tuning**, **HFRL** (Human Feedback Reinforcement Learning), **Prompt Engineering**, **RAG** (Retrieval-Augmented Generation), and **Control Codes** that can be used to _guide_ GPTs to generate more _meaningful_ outputs, but these are not _learning_ processes; they are _tweaking_ processes
  - Ultimately, there's no current technique that can make a GPT actually "understand" and "reason" about the **inputs** and **outputs** they're dealing with, as a human mind would do
- Sensitivity to input
  - GPTs are very sensitive to the **inputs** they process, and they can generate very different **outputs** based on small changes in input _phrasing_ (or **prompting**)
  - Asking the same question in different word orders, or even using different languages, can lead to very different outputs, sometimes even conflicting or entirely unrelated
    - For example, asking "_What is the capital of France?_" and "_France's capital is_" can lead to very different outputs, depending on the model's training
    - This happens because even if the **semantic** meaning of these questions is almost the same in these two phrasings, the **syntactic** meaning is different, and GPTs are very sensitive to these differences
    - In practice, every _character_ difference in a **prompt** can move the coordinates in the _text space_ function that computes the output value over that input, regardless of whether that character changes the _semantic_ meaning of the **prompt**
      - As a loose metaphor, we could imagine a function `f(x)` that gives possibly very different outputs if we give `x = 1`, `x = 1.00000001`, or `x = 1.000000000000...001`, even if the **semantic** meaning of these inputs is almost the same
- Inconsistency
  - GPTs may be inconsistent in generating outputs for the same inputs, even if the **inputs** are identical
  - Since most operations in the **inference** process are based on **probabilities** and **patterns** observed in the training data, the **outputs** can be very different for the same **inputs** based on the **randomness** of the **sampling** process
  - The results can be more or less deterministic based on the model **configurations** and **parameters** used in the **inference** process
    - There are parameters like **Temperature**, **Top-K**, **Top-P**, and **Nucleus Sampling** that can be used to **control** the **randomness** of the **sampling** process
  - A model operating with **greedy search** will generate the same outputs for the same input, while a model operating with **random sampling** (or **temperature** sampling) might often generate different outputs for the same input
- Incorrect or Nonsensical Outputs (and hallucinations)
  - As we have seen in previous lessons, GPTs (as we have nowadays) lack on **neural symbolic** reasoning capabilities, thus they are simply outputting whatever is most probable according to the **patterns** and **probabilities** observed in the training data
  - GPTs can generate incorrect or nonsensical outputs based on the **inputs** they process, especially when the **inputs** are very _ambiguous_ or _open-ended_
  - This is not necessarily a defect of GPTs, but a consequence of the **patterns** and **probabilities** observed in the training data
    - For example, a GPT trained on human anatomy may answer the question "How many eyes does a spider have?" with "Two" because it has learned from the training data that most subjects have two eyes, and it might not be clear in the training data that spiders are not humans
      - Even though this information is incorrect (according to the common definition of spiders), it might be _statistically_ correct to associate "number of eyes (of a living thing)" with "two" based on the training data of that model
  - This is very common with ambiguous topics, such as person names, locations, historical events, dates, quantities, and other subjects where there may be many different _correct_ answers based on the **context** of the **inputs**
    - The question "Who is James?" may vary greatly depending on the **context** of the question and the training data of the model
      - For example, if the training data contains many texts about James Bond, the GPT may answer "James Bond" to this question, even if the question is about another James
  - The term [Hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)) is often used in AI to describe these incorrect or nonsensical outputs because they are generated based on **patterns** and **probabilities** observed in the training data, not based on "understanding" or "reasoning" about the **inputs** and **outputs** they're processing
    - These hallucinations can be dangerous, as GPTs may provide very _convincing_ and _realistic_ outputs based on **patterns** and **probabilities** observed in the training data, even if these outputs are completely incorrect or nonsensical
- Bias and toxicity
  - GPTs can generate outputs that are biased and toxic based on the **inputs** they process, especially when the information in the **training data** is biased and toxic
  - Techniques like **Fine Tuning** and **HFRL** (Human Feedback Reinforcement Learning) can help mitigate this effect by tweaking the responses to be as _helpful_, _harmless_, and _honest_ as possible
    - However, even human feedback and direct oversight of the training data cannot guarantee that GPTs will generate unbiased and non-toxic outputs, as the very definitions of "helpfulness", "harmlessness", and "honesty" can be highly subjective, depending on various social, cultural, contextual, and personal factors
  - Current AI models (as of mid 2024) have a strong inclination to generate outputs based on _mild-leftist_ biases, as we have in the conclusions of studies like the [Political Preferences of LLMs](https://arxiv.org/pdf/2402.01789) paper by David Rozado and the [Tracking AI](https://trackingai.org/) website currently maintained by Maxim Lott
- No (native) access to real-time information
  - Since GPTs don't "learn" or "improve" when processing inferences, they can't access any information that occurs after their training, unless explicitly provided with this information in the **inputs** or through techniques like **RAG** (Retrieval-Augmented Generation) or similar methods
  - This means that GPTs can't provide real-time information about anything, such as news, events, weather, stock prices, sports results, etc.
    - It is possible to provide GPTs with real-time information, but this is separate from the training phase and is not a true learning process; it's simply passing data to the model with the expectation that it will process and return it correctly

### Practical Exercises with GPTs

We can use the Transformers Text Generation Pipeline to run GPTs for text-to-text tasks, similar to what we did in Lesson 2.

- Practical exercises
  - Exercise 3: Interacting with the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) application to understand the inner components of the Transformer architecture
  - Exercise 4: Using the [Text Generation Pipeline](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextGenerationPipeline) to run the GPT-2 model for text-to-text tasks with Hugging Face's Transformers library using an [example notebook](https://colab.research.google.com/github/dipanjanS/nlp_workshop_odsc19/blob/master/Module05%20-%20NLP%20Applications/Project08%20-%20Text%20Generation%20with%20Transformers.ipynb)
    - Fix the "Paragraph Generation with GPT-2" example by replacing the last cell with this: `!python pytorch-transformers/examples/pytorch/text-generation/run_generation.py --model_type=gpt2 --length=500 --model_name_or_path=gpt2`

## Next Steps

- Setting up a Python development environment
- Using OpenAI APIs
- How to use GPTs with API calls
- Handling requests
- Streaming responses
- Model configurations
- Model comparisons
- Prompt engineering
- Weekend project
