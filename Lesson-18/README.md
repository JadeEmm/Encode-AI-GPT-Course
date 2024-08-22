# Lesson 18: AI Agents & Introduction to Computer Vision

In this lesson, we will explore how to build AI agents using open-source tools and leverage Large Language Models (LLMs) to power the reasoning process behind agent task execution.

We will also examine how AI Agents can be integrated into Retrieval-Augmented Generation (RAG) pipelines to automate decision-making about information retrieval from various indexes and its application to the generation task at hand.

By the end of this lesson, we will begin our introduction to computer vision and discuss how it differs from traditional Natural Language Processing (NLP) models.

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
- Installation of Node.js on your device
  - [Node.js](https://nodejs.org/en/download/)
- Proficiency in using `npm` and `npx` commands
  - Documentation: [npm](https://docs.npmjs.com/)
  - Documentation: [npx](https://www.npmjs.com/package/npx)
- Proficiency in using `npm install` and managing the `node_modules` folder
  - Documentation: [npm install](https://docs.npmjs.com/cli/v10/commands/npm-install)
- Installation of `git` CLI on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency in using `git` commands to clone repositories
  - Documentation: [Git](https://git-scm.com/doc)

## Review of Lesson 17

- AI Agents
- Task planning
- Goals
- Types of agents
- Applications and examples
- Setting up an AI agent

## Building a Simple AI Agent Program

- Defining tools
- Using a Query Engine for processing tasks
- Implementing Query Transformations
- Task planning
- Handling steps
- Evaluating goals

- Practical exercise
  - Exercise 1: Implement a simple AI agent using the [Agent Tools](https://docs.llamaindex.ai/en/stable/use_cases/agents/) from [LlamaIndex](https://llamaindex.ai)

## Agentic RAG

- Retrieval Agents
  - File search functions
  - Web search functions
  - Chunking data
  - Embeddings
  - Vector search
- Integrating Query Engines into Agent tasks
- Multi-tool invocation
- Implementing Query Transformations
  - Decision-making for information retrieval
- Multi-step queries

**Practical exercise:**

- Exercise 2: Implement a [Query Engine Tool](https://docs.llamaindex.ai/en/stable/understanding/agent/rag_agent/) for an AI Agent using sample data

## Introduction to Computer Vision

- Working with images
  - NLP tasks benefit from LLMs by relating the probabilities of textual data in the prompt to the probabilities of textual chunks being generated as responses
  - Visual (non-verbal) data requires a different approach for correlating visual elements to conceptual expressions in text
    - This necessitates a distinct methodology for relating visual elements in an image to conceptual expressions in text
- Examples of applications
  - Facial recognition: Identifying or verifying a person from their face
  - Pose estimation: Detecting the position and orientation of human body parts
  - Optical Character Recognition (OCR): Extracting text from images
  - Image captioning: Generating textual descriptions of images
  - Object tracking: Following the movement of objects across video frames
  - Depth estimation: Predicting the depth of objects in 2D images
  - Anomaly detection: Identifying unusual patterns or objects in images
  - Document analysis: Extracting structured information from scanned documents
  - Fraud detection: Identifying suspicious features in images or documents
  - Traffic control: Monitoring and managing traffic flow

## Overview of Computer Vision

- Early approaches
  - Before the widespread adoption of deep learning for computer vision tasks, researchers utilized traditional algorithms to process image elements:
    - **SIFT** (Scale-Invariant Feature Transform):
      - Detects and describes local features in images
      - Robust to scaling, rotation, and illumination changes
    - **SURF** (Speeded Up Robust Features):
      - Faster alternative to SIFT
      - Uses box filters and integral images for feature detection
    - **HOG** (Histogram of Oriented Gradients):
      - Counts occurrences of gradient orientations in localized portions of an image
      - Commonly used for object detection
    - **Canny Edge Detection**:
      - Identifies edges in images using a multi-stage algorithm
      - Widely used for feature detection and image segmentation
    - **Hough Transform**:
      - Detects lines, circles, and other shapes in images
      - Useful for identifying geometric structures
    - **Haar Cascades**:
      - Uses rectangular features to detect objects in images
      - Commonly used for face detection
- Model evolution
  - The rise of deep learning algorithms and advanced AI models for Computer Vision has largely supplanted many traditional algorithms, enabling significantly more complex tasks with greater accuracy and performance
  - This rapid evolution has been driven by investments in industries such as space exploration, climate research, military applications, autonomous vehicles, data security, social networks, and mobile devices (e.g., facial recognition for device unlocking)
- Convolutional Neural Networks (**CNNs**) for image processing
  - CNNs were the first models to achieve state-of-the-art results on computer vision tasks before the advent of transformer-based models
- Vision Transformer (**ViT**)
  - Model that divides images into fixed-size patches and linearly embeds them (flattened and projected into a higher-dimensional space) along with positional encodings to retain information about their original location in the image
    - These embeddings can then be processed by the transformer encoder to perform tasks similar to text generation models or even image generation
- Detection Transformers (**DETR**)
  - This model uses a Convolutional Neural Network backbone to extract a feature map from the input image, which is then flattened and passed through a series of transformer encoders and decoders to produce the final output
  - The encoder processes the feature map to capture relationships between different parts of the image, while the decoder takes in a fixed number of learned positional embeddings called "object queries" and decodes them in parallel to predict object classes and their bounding boxes
- Contrastive Language-Image Pretraining (**CLIP**)
  - Released by OpenAI in 2021, this model significantly improved the state-of-the-art in image classification by combining a vision transformer with a language transformer
  - This model is trained to be multi-modal by using two encoder models that can relate tokens to both images and text
    - The model is trained to relate words to the token representations of images depicting what these words describe
      - For example, the word "dog" would be more closely related to the token representation of an image of a dog than to that of a cat, while the word "cat" would be more closely related to the token representation of an image of a cat than to that of a dog
    - This approach enables the model to relate images and texts bidirectionally
- Masked Autoencoders for Visual Learning (**MAE**)
  - Introduced by Facebook AI Research in 2021, MAE is a self-supervised learning approach for vision transformers
  - It operates by masking random patches of the input image and then attempting to reconstruct the missing pixels
  - This pretraining technique has proven highly effective for downstream tasks such as image classification and object detection
- Segment Anything Model (**SAM**)
  - Introduced by Meta AI in 2023, SAM is a promptable segmentation model capable of generating segmentation masks for any object in an image based on textual prompts or by selecting a region of an image or video frame as a starting point
  - It can generate segmentation masks for any object in an image based on various types of prompts (points, boxes, or text)
  - SAM demonstrates impressive zero-shot generalization capabilities across a wide range of segmentation tasks
    - As of mid-2024, the [SAM 2](https://ai.meta.com/sam2/) model is one of the most capable and performant segmentation models publicly available

## Next Steps

- Overview of Computer Vision models
- Computer Vision tasks
- Using Vision Transformers (ViT)
- Object detection
- Object classification
