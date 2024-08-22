# Lesson 01: Introduction to AI

In this lesson, we will introduce the basic concepts of AI, its capabilities and limitations, while debunking some myths and misconceptions about it. We will also discuss practical uses of AI with an example of using ChatGPT for performing text-to-text tasks.

## Prerequisites

- Nothing special
  - Just be curious and willing to learn
  - (Optional) Have a good source of hydration and a comfortable place to sit
    - And maybe some snacks?

## What is AI?

- AI before November 30, 2022
  - First formal mention of AI was in [1956 by John McCarthy](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth)
  - First decades of AI focused on rule-based systems
    - AI was seen as a way to **automate** tasks that required human intelligence
  - By early 2000s, AI applications became more common, especially in digital environments
    - AIs for marketing, customer service, and other digital tasks
    - AIs for search engines, recommendation systems, and other digital services
    - AIs for games, simulations, and other digital entertainment
- November 30, 2022 -> ChatGPT launch date
- AI now
  - With ChatGPT (and all similar generative AI tools that followed), AI has entered an age of **mass adoption**
  - AI is now used to **create** content, not just to **automate** tasks
  - AI is now used to (attempt to) _"understand"_ content, not just to **process** data
  - AI is now used to **generate** brand new content, not just to **recommend** or slightly **modify** existing content

## Learning AI

- Common prerequisites for working with AI
  - Basic programming skills
    - Python
    - Python development tools
    - Libraries and dependencies
    - Defining and calling functions
    - Classes, variables, and objects
    - Dictionaries, lists, and sets
    - Loops and conditionals
  - Basic understanding of statistics
    - Mean, median, mode, and outliers
    - Standard deviation
    - Probability
    - Distributions
  - Basic understanding of algebra
    - Variables, coefficients, and functions
    - Linear equations
    - Logarithms and logarithmic equations
    - Exponential equations
    - Matrix operations
    - Sigmoid functions
  - Basic understanding of calculus
    - Derivatives
    - Partial derivatives and gradients
    - Integrals
    - Limits
    - Sequences and series

## Learning Practical AI Applications

- Prerequisites
  - A computer or similar device
  - Internet connection
  - Dedication
  - Time
- Everything else will be covered as we progress through this bootcamp

## AI Bootcamp

- Example of AI application: [ChatGPT](https://chat.openai.com/)

> Did someone program the application to understand and generate text for each single word in each single language?
>
> > No, the application was **trained** to _"understand"_ and generate text

- How to **train** an application?
  - Applications are pieces of **software** that run on a **computer**
  - How do we **train** a **computer**?
- Machine learning

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." [Tom M Mitchell et al](https://www.cs.cmu.edu/~tom/mlbook.html)

- Does it mean that the computer is _"learning"_?
- Does it mean that the computer is _"thinking"_?
- Does it mean that the computer is _"conscious"_?

## AI Tasks

- Programming a system or application to solve a specific task can take a lot of time and effort, depending on the complexity of the task and the number of edge cases that need to be considered
  - Imagine programming a system to translate text from one language to another by looking up words in a dictionary and applying grammar rules, comparing contexts, and considering idiomatic expressions for every single word in every single variation of their usage
    - Such applications are simply not feasible to be programmed by hand, not even by a huge team of programmers
- For these situations, **AI Models** can be **trained** for statistically solving the task without necessarily handling the actual _"reasoning"_ or _"understanding"_ of the task
  - The **training** process is done by **feeding** the model with **examples** of the task and the **correct** answers
  - The **model** then _"learns"_ the **patterns** and **rules** that **statistically** solve the task
  - The **model** can then **predict** the **correct** answer for new **examples** that it has never seen before

### Examples of AI Tasks

- **Natural Language Processing (NLP)**
  - Question answering
  - Feature extraction
  - Text classification (e.g., Sentiment Analysis)
  - Text generation (e.g., Text Summarization and Text Completions)
  - Fill-Mask
  - Translation
  - Zero-shot classification
- **Computer Vision (CV)**
  - Image/video classification
  - Object detection
  - Image/video segmentation
  - Image/video generation
  - Image/video captioning
  - Depth estimation
  - Image/video feature extraction
  - Mask generation
  - Zero-shot classification
  - Zero-shot detection
- **Image Generation**
  - Text-to-image
  - Image-to-image
  - Image-to-3D
  - Text-to-3D
  - Text-to-video
  - Image-to-video
  - Video-to-video
  - Video-to-3D-video
  - Text-to-3D-video
- **Audio processing**
  - Speech recognition
  - Speech generation
  - Audio classification
  - Audio generation
  - Audio-to-text generation
  - Text-to-audio generation
  - Audio-to-image generation
  - Audio-to-video generation
- **Multi-modal tasks**
  - Text-image-to-text generation
  - Text-image-audio-to-text generation
  - Text-image-audio-video-to-text generation
  - Text-to-text-image generation
  - Text-to-text-audio generation
  - Text-to-text-image-audio generation
  - Text-to-text-image-audio-video generation
  - Document question answering
  - Visual question answering

## Capabilities and Limitations of AI

- [Stochastic parrot](https://en.wikipedia.org/wiki/Stochastic_parrots)
  - A critique view of current LLMs (large language models)
    - LLMs are limited by the data they are trained by and are simply stochastically repeating contents of datasets
    - Because they are just "making up" outputs based on training data, LLMs do not _understand_ if they are saying something incorrect or inappropriate
- [The Chinese Room](https://en.wikipedia.org/wiki/Chinese_room) philosophical experiment presented by John Searle in 1980
  - The notion of machine _understanding_
  - The implementation of syntax alone would constitute semantics?
  - A _simulation_ of mentality can be considered as a **replication** of mentality?
- Can AI truly "understand" language?
  - What is, indeed, "understanding"?
    - [Aristotle. (350 BCE). Metaphysics.](https://classics.mit.edu/Aristotle/metaphysics.html): Knowledge of causes
    - [Locke, J. (1690). An Essay Concerning Human Understanding.](https://www.gutenberg.org/files/10615/10615-h/10615-h.htm): Perception of agreement or disagreement of ideas
    - [Dilthey, W. (1900). The Rise of Hermeneutics.](https://www.degruyter.com/document/doi/10.1515/9780691188706-006/html?lang=en): Interpretive process of making meaning
    - [Ryle, G. (1949). The Concept of Mind.](https://archive.org/details/conceptofmind0000ryle): Ability to apply knowledge in various contexts
    - [Chalmers, D. (1996). The Conscious Mind.](https://personal.lse.ac.uk/ROBERT49/teaching/ph103/pdf/Chalmers_The_Conscious_Mind.pdf): Functional role in cognitive system
  - As of our current knowledge, to "understand" means:
    - To grasp the meaning or significance of information
      - Example: Recognizing that "It's raining cats and dogs" is an idiom, not a literal statement
      - Example: Interpreting a graph showing climate change trends over time
    - To process and interpret data in a meaningful way
      - Example: Analyzing sales figures to identify seasonal patterns
      - Example: Interpreting medical test results to diagnose a condition
    - To apply knowledge in various contexts appropriately
      - Example: Using algebra skills to calculate a tip at a restaurant
    - To make logical inferences and connections
      - Example: Deducing that a suspect is left-handed based on crime scene evidence
      - Example: Connecting historical events to understand their impact on current geopolitics
      - Example: Understanding the role of technology advancements in shaping the social and political changes in the past centuries
    - To recognize patterns and relationships
      - Example: Identifying the Fibonacci sequence in nature
      - Example: Noticing correlations between diet and health outcomes in a study
    - To adapt knowledge to new situations
      - Example: Using cooking skills to improvise a meal with limited ingredients
    - To communicate ideas effectively
      - Example: Explaining a complex scientific concept to a child
    - To solve problems using acquired knowledge
      - Example: Troubleshooting a malfunctioning computer using technical expertise
      - Example: Resolving a conflict between coworkers using conflict resolution strategies
    - To demonstrate awareness of one's own thought processes
      - Example: Recognizing and correcting one's own biases in decision-making
    - To exhibit creativity in applying understanding
      - Example: Combining ingredients in novel ways to create a new recipe
- The current capabilities of AI models
  - Limited to "statistical" reasoning
  - Infer answers based on patterns and correlations in data
    - Often the correct answer is very similar to wrong answers (hallucinations)
  - The architectures of the current most popular models (as of mid 2024) are not able to process [neuro-symbolic](https://en.wikipedia.org/wiki/Neuro-symbolic_AI) parameters
- **Weak AI** vs **Strong AI**
  - Weak AI: Designed for specific tasks, lacks general intelligence
  - Strong AI: Hypothetical AI with human-like general intelligence
- Concept of **AGI** (Artificial General Intelligence)
  - AI with human-level cognitive abilities across various domains
  - Ability to transfer knowledge between different tasks
  - Potential to surpass human intelligence in many areas
- Symbol processing
  - Able to _reason_ beyond the connectionist approaches in current popular AI models
    - Manipulation of symbolic representations
    - Logical inference and rule-based reasoning
    - Explicit representation of knowledge through linking symbols
    - Formal manipulation of symbols to derive conclusions
    - Ability to handle abstract concepts and relationships
- Meta consciousness
  - Claude-3 Opus [needle-in-the-haystack experiment](https://medium.com/@ignacio.serranofigueroa/on-the-verge-of-agi-97556c35692e)
    - Impression of consciousness due to the **instruction following** fine tuning
  - Hermes 3 405B [empty system prompt response](https://venturebeat.com/ai/meet-hermes-3-the-powerful-new-open-source-ai-model-that-has-existential-crises/)
    - Impression of consciousness due to the amount of similar data present in the training set (possibly role-playing game texts)

## Practical Uses of Generative AI

- Dealing with text inputs
  - What is **Natural Language Processing (NLP)**
  - Much more than just **replying** to word commands
    - Example: [Zork](https://en.wikipedia.org/wiki/Zork) text input processor
  - **NLP** AI Models are able to process text inputs by relating the **concepts** related to the textual inputs with the most probable **concepts** in the training set
    - Ambiguity in textual definitions
    - Contextual variations
    - Cultural variations
    - Semantic variations
- Dealing with image inputs
  - What is **Computer Vision (CV)**
  - Dealing with elements inside an image
  - Dealing with the _"meaning"_ of an image
- Dealing with audio inputs
  - What is **Automatic Speech Recognition (ASR)**
  - Dealing with spoken commands
  - Categorizing noises and sounds
  - Translating speech/audio to text/data elements
- Generating **text outputs**
- Generating **image outputs**
- Generating **audio/speech outputs**
- Generating **actions**
  - API calls
  - Integrations
    - Interacting with the real world through robotics

## Getting Started with Generative AI for Text-to-Text Tasks

- Using the [OpenAI Platform](https://platform.openai.com/)
  - [Docs](https://platform.openai.com/docs/introduction)
- Practical exercises
  - Exercise 1: Using the [OpenAI Chat Playground](./exercises/00-OpenAI-Chat-Playground.md)
- Does it sound like a _"real person"_?
- Does it sound _"intelligent"_?

## Next Steps

- What is **Machine Learning**
- What is happening when we submit a prompt to **ChatGPT**
