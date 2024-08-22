# Lesson 19: AI Image Processing

In this lesson, we'll explore the extraction of information from images using Computer Vision models. We'll examine how these models function and their application to various computer vision tasks.

We'll review how the versatile Transformers architecture, previously discussed for text and image generation, can be applied to image processing tasks.

We'll experiment with Transformers to perform zero-shot detection and classification tasks to gain a deeper "understanding" of computer vision capabilities.

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
- Proficiency with `git` commands for cloning repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Account at [Google Colab](https://colab.research.google.com)

## Review of Lesson 18

- Agentic RAG
- LlamaIndex utilization
- Computer Vision fundamentals
- Image processing algorithms and models

## Computer Vision Models

- Computer Vision (CV)
  - CV is a field of artificial intelligence that enables computers to interpret and understand visual information from the world
  - It involves developing algorithms and models that can process, analyze, and extract meaningful information from digital images and videos
  - CV has applications in various domains, including autonomous vehicles, facial recognition, medical imaging, and robotics

- Model Training
  - Model training in CV involves feeding large datasets of labeled images or videos to machine learning algorithms
  - The algorithms learn to recognize patterns, features, and relationships within the visual data
  - During training, the model adjusts its internal parameters to minimize the difference between its predictions and the actual labels
  - This process typically involves techniques such as backpropagation and gradient descent to optimize the model's performance

- Inference
  - Inference in CV refers to the process of using a trained model to make predictions or decisions on new, unseen data
  - During inference, the model applies the knowledge it gained during training to analyze and interpret new images or videos
  - This stage is where the practical applications of CV models come into play, such as identifying objects in real-time video streams or classifying medical images

- Image Processing Tasks
  - Image processing tasks in CV often involve manipulating or analyzing images to extract useful information or enhance their quality
  - These tasks can include operations like filtering, edge detection, color correction, and image segmentation
  - In the context of modern AI models, these tasks can be initiated through prompts or instructions given to the model, allowing for more flexible and dynamic image processing capabilities

- Using Transformers for CV
  1. **Image Classification**
     - A process where an algorithm is trained to recognize and categorize images into predefined classes or labels
     - Example: Classifying an image as a "cat," "dog," or "car"

  2. **Image Segmentation**
     - Involves dividing an image into multiple segments or regions based on specific criteria, such as objects, boundaries, or pixel similarities

  3. **Video Classification**
     - Similar to image classification, but analyzes video frames to categorize the entire video or its segments into predefined classes
     - Considers temporal information and may involve recognizing actions, events, or behaviors over time
     - Example: Classifying a video as "sports," "news," or "entertainment"

  4. **Object Detection**
     - The task of identifying and localizing specific objects within an image or video frame
     - Involves drawing bounding boxes around the detected objects and assigning them corresponding labels
     - Example: Detecting and localizing cars, pedestrians, and traffic signs in an autonomous driving scenario

  5. **Zero-shot Detection**
     - Refers to the ability of a model to recognize and locate objects it has never seen during training
     - Achieved by using prior knowledge, such as the semantic relationship between objects or attributes, to infer the presence of unseen classes without explicit examples

  6. **Zero-shot Classification**
     - Similar to zero-shot detection but focuses on assigning labels to images or videos without having seen training examples of those specific classes
     - Relies on utilizing semantic information or descriptions of the unseen classes to make predictions
     - Example: Classifying an image as a "giraffe" based on its description, even though the model has never been trained on giraffe images

  7. **Single-shot and Few-shot Detection and Classification**
     - Refers to the ability of a model to detect or classify objects with minimal training examples
     - Single-shot aims to perform these tasks with a single training example per class, while few-shot uses a few examples per class
     - Useful for scenarios where collecting large amounts of labeled data is challenging or expensive

## Running Computer Vision Models

- The [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library provides utilities for running pre-trained models for various computer vision tasks
- Example tutorials and documentation:
  - [Quick Demo for Vision Transformers](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb) (`ViT`) for image classification
  - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (`OWL-ViT`) model with [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/model_doc/owlvit) for [Zero-shot object detection](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb)

## Running Computer Vision Tasks with Hugging Face Transformers

- Computer Vision [pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines#computer-vision)
- [Zero-Shot](https://huggingface.co/tasks/zero-shot-classification) tasks for [Image Classification](https://huggingface.co/tasks/zero-shot-image-classification) and [Object Detection](https://huggingface.co/tasks/zero-shot-object-detection)
- Image manipulation in Python using the [Pillow](https://pypi.org/project/pillow/) imaging library

  - Drawing on images
  - Creating rectangle boxes based on coordinates
  - Adding text at specific coordinates

- Practical exercises:
  1. Run a [Zero-shot object detection pipeline](./exercises/00-Object-Detection.md)
     - Code [ObjectDetection.py](./examples/ObjectDetection.py) and test the model with a sample image
  2. Run a [Zero-shot image classification pipeline](./exercises/01-Image-Classification.md)
     - Code [ImageClassification.py](./examples/ImageClassification.py) and run it with a file argument to output an image description

## Next Steps

- Experiment with additional Computer Vision models
- Explore multimodal models incorporating Computer Vision
- Define AI World Scope
- Integrate computer vision into applications
