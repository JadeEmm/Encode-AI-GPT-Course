# Lesson 14: Extending the Capabilities of GPTs

In the previous lesson, we explored GPT model files and their construction through neural network training on datasets.

This lesson focuses on expanding these models' capabilities using techniques such as Low-Rank Adaptations (LoRAs) and creating assistants.

We'll build upon our knowledge of training and fine-tuning models with datasets by using Text Generation WebUI to train a LoRA for GPT-2-medium and evaluate its performance.

Next, we'll compare various base and fine-tuned models to observe how they generate text with different styles and topics. The Vercel AI Playground will be our tool for side-by-side model comparisons using sample prompts.

Finally, we'll utilize the OpenAI Platform to create and test an assistant, manage threads and messages, and implement augmentation jobs. We'll explore the practical applications of assistants and how they can consistently respond to structured text generation tasks across multiple threads from various users.

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
- Proficiency with `git` commands for repository cloning
  - Documentation: [Git](https://git-scm.com/doc)
- Account with [Vercel](https://vercel.com/) to access the [Vercel AI Playground](https://sdk.vercel.ai/)
- Account at [OpenAI Platform](https://platform.openai.com/)
  - For running API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 13

- Types of models
- Hardware requirements and model optimization for specific hardware
- Overview of GPT training
- Fine-tuning GPT-2

## Low-Rank Adaptations (LoRAs)

- Understanding LoRAs
  - LoRAs (Low-Rank Adaptation) are a technique for fine-tuning large language models with minimal additional memory requirements
    - LoRA is a parameter-efficient fine-tuning technique for LLMs. It freezes the weights of the LLM and injects trainable rank-decomposition matrices.
  - LoRAs adapt a model to a specific task without fine-tuning all parameters, which can be memory-intensive and computationally expensive
    - A LoRA modifies the weights of the pre-trained model in a low-rank subspace, rather than updating all its parameters
    - It's a form of parameter-efficient transfer learning, focusing on adapting pre-trained models to new tasks with minimal changes to the original model
      - Although the total number of parameters increases (due to added LoRA layers), the memory footprint reduces because the number of trainable parameters decreases
  - Low-rank matrices modify the behavior of attention and feed-forward layers within the transformer model
  - Currently (as of mid-2024), Text Generation WebUI only allows using `llama.cpp`, `ctransformers`, and `AutoAWQ` loaders when loading LoRAs
- Efficiently training models with LoRAs
  - More model parameters require more memory for training and increase computational costs for gradient calculations
  - LoRAs enable model adaptation to specific tasks without fine-tuning all parameters, reducing memory and computational requirements
    - They are effective for fine-tuning with smaller datasets and overcoming memory and computation limitations
    - This accelerates the fine-tuning process by reducing the number of trainable parameters
  - After loading the model and applying the LoRA, there's no additional inference latency as the LoRA is already integrated into the model
- Limitations of LoRAs
  - Despite efficient training, LoRAs can be **challenging** to effectively alter model behavior in a controlled manner
  - They may lead to unintended changes in model behavior, such as hallucinations or nonsensical outputs
  - LoRAs are typically used in narrow scopes to facilitate mapping and measuring the impact of changes in model behavior
- Reusing LoRAs with other models
  - Once created, a LoRA is a _separate_ entity from the original model and can be reused with other models without retraining
  - This is achieved by loading the LoRA in the model's configuration and using it to modify the model's behavior
  - The usefulness of a LoRA trained on model A when applied to model B depends heavily on the specific task and the overall difference between the models
    - In most cases, a LoRA would likely have random effects on models different from those it was trained for, even with similar architectures and weights
- Examples of common LoRA applications
  - Instruction-following LoRAs (e.g., [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) for LLaMA)
  - Language-specific LoRAs (e.g., [Chinese-Alpaca-LoRA](https://huggingface.co/hfl/chinese-alpaca-lora-7b))
  - Task-specific LoRAs (e.g., for code generation, medical diagnosis, or legal document analysis)
  - Style-adaptation LoRAs (e.g., for creative writing in specific author styles)
  - Domain-specific LoRAs (e.g., for scientific literature or financial analysis)

## Creating a LoRA for GPT-2-medium

- Practical exercise
  - Exercise 1: [Train GPT-2 Medium](./exercises/00-Train-GPT2-Medium.md) to generate text with a text file dataset using the Text Generation WebUI [Training Tab](https://github.com/oobabooga/text-generation-webui/wiki/05-%E2%80%90-Training-Tab)

## Comparing Text Generation Models

- The [Vercel AI Playground](https://sdk.vercel.ai/)
- Comparing models using prompts
- Various versions of OpenAI GPT
- Different iterations of [Llama](https://llama.meta.com/)
- Evaluating [Claude](https://claude.ai/)
- Assessing [Mistral](https://mistral.ai/)
- Examining [Gemma](https://blog.google/technology/developers/gemma-open-models/)

### Comparing Text Generation Quality

- Practical exercise
  - Exercise 2: [Using Vercel AI Playground](./exercises/01-Using-Vercel-AI-Playground.md) to compare different text generation models

## Code Generation Models

- Training and fine-tuning techniques for code generation
- Instruction-following capabilities
- Retrieval and context evaluation methodologies
- Optimizing behavior through parameter adjustments and constraints

### Code Generation Examples

- [CodeLlama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)
- [Copilot](https://copilot.github.com/)
- [Cursor](https://cursor.com/)
- [Replit AI](https://replit.com/ai)
- [Q Developer](https://aws.amazon.com/q/developer/)
- [Cody](https://sourcegraph.com/docs)
- [Devin](https://www.cognition-labs.com/blog), the AI software engineer

## Addressing the Limitations of GPTs

- Capabilities of GPTs "out of the box"
- Providing instructions to GPTs
- Utilizing GPTs for code generation
- Managing context and relevant information

## Using Assistants

- Definition of a GPT assistant
  - AI Assistants have existed for a long time, typically as chatbots programmed to respond to specific prompts or commands in a structured manner
  - GPTs enhance these assistants, making them more flexible and capable of handling a wider range of tasks, sometimes even beyond their explicit training
  - This applies to both processing user input to determine intent and executing tasks according to that intent
  - Unlike standard chat or instruction tasks in a GPT, an assistant can be pre-configured and reused across multiple threads and messages for a consistent user experience
    - These assistants can be enhanced with functions that expand their capabilities, such as retrieving web information, accessing files, or making API calls
- Assistant roles and personal instructions
  - An assistant can be configured to respond to specific prompts or commands in a structured way, ensuring consistent and helpful responses
  - Instead of relying on the end user to provide the correct prompt, the assistant can be programmed to "understand" the intent from the user's input and execute tasks accordingly
    - It's possible to program the assistant to request more information if the input lacks clarity, based on the "clarity" criteria most associated with the terms used in the input according to the training data
  - Once configured, assistants can respond to multiple threads for different users without mixing contexts
    - This is achieved by programming the assistant to manage the context of each thread and message, as well as the relevant information for each thread
    - This allows each user to have a personalized experience with the assistant, based on their specific messages and threads
- Managing messages within a thread
  - The assistant can be programmed to manage messages within a thread, tracking context and relevant information without mixing them up
- Using files within threads
  - Assistants can be programmed to use files within threads, retrieving information and generating responses based on file content
  - Rather than including entire file contents in the prompt context, the assistant can be programmed to _retrieve_ information from the file as needed, _augmenting_ the generation process with the most relevant data fragments for the current prompt
    - This allows the assistant to maintain a manageable context size even when processing large files
- Processing chat requests
  - Assistants can be programmed to process chat requests, generating responses based on message content and thread context
  - Instead of generating responses based solely on the last message, the assistant can be programmed to consider the entire thread context, including previous messages
    - This enables the assistant to generate more consistent and relevant responses based on the entire conversation within the thread
- Processing instructions
  - Assistants can be programmed to process end-user instructions and combine them with pre-programmed instructions to generate responses and actions
- Generating code
  - A common use for such assistants is generating code snippets based on user input, aiding in script completion or finding solutions to specific programming problems in user-provided files
- Generating images
  - Another common use is generating images based on user input by triggering specific functions that request image generation from capable models
- Generating files
  - Assistants can be programmed to generate files for user download
    - For example, the assistant could automatically provide a `pdf` version of the chat or a `csv` table summarizing requested financial data
- Retrieving information from the web
  - _Retrieval_ functions are not limited to user-provided files; they can also be used to retrieve web information, such as searching for specific terms or topics on the internet and providing results to the user

## Building and Testing an Assistant with OpenAI APIs

OpenAI [Assistants](https://platform.openai.com/docs/assistants/overview) offer a quick and easy way to set up and test custom-made assistants powered by GPT models. These assistants can be programmed to respond to specific prompts or commands in a structured manner, providing consistent and helpful responses to users.
OpenAI Assistants are currently equipped with a robust set of [Useful Tools](https://platform.openai.com/docs/assistants/tools) for tasks such as code interpretation, knowledge retrieval, function execution (e.g., retrieving weather information for a city), and more.

- Practical exercises
  - Exercise 1: [Creating a simple assistant](./exercises/01-Create-Assistant.md) with instructions to test system prompts for assistants
  - Exercise 2: [Creating a message thread](./exercises/02-Create-Message-Thread.md) with the assistant to test how assistants handle threads and contexts

## Next Steps

- Retrieval Augmented Generation (RAG)
- Using OpenAI Assistants with RAG
- Building a chat application with OpenAI Assistants
- Implementing RAG in the chat application
