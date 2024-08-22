# Lesson 20: Computer Vision

In our previous lesson, we explored how Computer Vision models can be applied to various image processing tasks.

This lesson delves deeper into the relationship between textual concepts (labels or captions) and visual concepts in images. We'll introduce the concept of **World Scope**, which offers a fascinating perspective on how information can be correlated in unprecedented ways when combined in sufficient volumes.

We'll examine the interplay between visual and textual concepts within multimodal models, laying the groundwork for our future study of image generation models capable of processing text-to-image tasks.

By the end of this lesson, we'll implement a multimodal chat application that leverages Computer Vision models to process images before using them to answer prompts.

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
- Basic knowledge of JavaScript programming language syntax
  - [JavaScript official tutorial](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/javascript/)
- Basic knowledge of TypeScript programming language syntax
  - [TypeScript official tutorial](https://www.typescriptlang.org/docs/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/typescript/)
- Account at [Google Colab](https://colab.research.google.com)

## Review of Lesson 19

- Overview of Computer Vision models
- Computer Vision tasks
- Vision Transformers

## Computer Vision with SAM 2

- The [SAM 2](https://ai.meta.com/blog/segment-anything-2/) is a state-of-the-art Computer Vision model for semantic segmentation of images and videos
- Key Features:
  - **Video Segmentation**: Enables object segmentation in videos, tracking across frames and handling occlusions
  - **Memory Mechanism**: Incorporates a memory encoder, bank, and attention module to store and utilize object information, enhancing user interaction throughout videos
  - **Streaming Architecture**: Processes video frames sequentially, allowing real-time segmentation of lengthy videos
  - **Multiple Mask Prediction**: Generates multiple possible masks for ambiguous images or video scenes
  - **Occlusion Prediction**: Improves handling of temporarily hidden or out-of-frame objects
  - **Enhanced Image Segmentation**: Outperforms the original SAM in image segmentation while excelling in video tasks
- Improvements:

  - Unified architecture for both image and video segmentation
  - Rapid video object segmentation
  - Versatile model capable of segmenting novel objects, adapting to unfamiliar visual contexts without retraining, and performing zero-shot segmentation on images containing objects outside its training set
  - Fine-tuning of segmentation results by inputting prompts for specific pixel areas
  - Superior performance across various image and video segmentation benchmarks

- Practical exercises:
  - Exercise 1: Run the [Segmenting Images with SAM 2](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb) Notebook
  - Exercise 2: Run the [Segmenting Videos with SAM 2](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-videos-with-sam-2.ipynb) Notebook

## Multimodal Models with Computer Vision

- Language alone is insufficient to provide comprehensive information about our universe to Machine Learning models

  - The [Experience Grounds Language](https://arxiv.org/abs/2004.10151) paper proposes an intriguing perspective on how these ML models "understand" the scope of our world

  > Language understanding research is held back by a failure to relate language to the physical world it describes and to the social interactions it facilitates. Despite the incredible effectiveness of language processing models to tackle tasks after being trained on text alone, successful linguistic communication relies on a shared experience of the world. It is this shared experience that makes utterances meaningful.

- The AI's "understanding" of concepts present in the training data can be categorized based on the types of "perceptions" fed to the model during training to correlate these concepts:

  - WS1: Corpora and Representations (Syntax and Semantics)
  - WS2: The Written World (Internet Data)
  - WS3: The Perceivable World (Sight and Sound)
  - WS4: Embodiment and Action (Physical Space and Interactions)
  - WS5: Social World (Cooperation)

- The [Contrastive Language-Image Pretraining](https://github.com/openai/CLIP) (CLIP) model from OpenAI is one of the [first models to combine text and vision](https://openai.com/index/clip/) to comprehend concepts in both text and image and even connect concepts between the two modalities
  - As it utilizes visual information in the training process itself, CLIP can be considered a [world scope three model](https://www.pinecone.io/learn/series/image-search/clip/)
- The relationship between Natural Language Processing (NLP) and Computer Vision (CV) concepts is made possible through the **Contrastive Pretraining** process applied to the CLIP model during training

- Practical exercises:
  - Exercise 3: Run a simple CLIP demo using the [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/model_doc/clip#usage-tips-and-example) library
  - Exercise 4: Explore CLIP with [Google Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
  - Exercise 5: Run the [CLIP-Explainability](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) Notebook

## Implementing Computer Vision Features in a Chat Application

- Run CLIP as a service on your device with [clip-as-a-service](https://pypi.org/project/clip-as-service/)
- Implement a simple chat application from the Multimodal Chat Application template
- Integrate the CLIP API to process image inputs in the chat application

## Weekend Project

To consolidate the knowledge acquired this week, students should complete the following project:

1. Create a GitHub repository for your project
2. Add all members of your group as collaborators
3. Create a README.md file with the description of your project
4. Create a new application from scratch using NextJS
5. Create a page with a single input field for the user to upload an image
   - Ideally, the user would upload a picture of an animal
6. Add a button to upload the image
7. Use a Computer Vision model to detect and classify the animal
   - The model should be able to detect at least 10 different animals of your choice
   - The model should return the name of the animal detected (classification)
8. Create an AI Agent that can find a page in [Wikipedia](https://wikipedia.org/) with the name of the animal, retrieve the description, and determine if the animal is dangerous
9. If the uploaded image contains an animal, pass the image to the AI Agent and await the answer
10. Display the answer on the page, indicating whether the animal in the picture is dangerous
11. Submit your project in the submission form

> You should find your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>
> > If you can't find your group, please contact the program manager through Discord or email

## Next Steps

- Image generation techniques
- Image generation AI models
- Stable diffusion
- Image generation applications
- OpenAI Image Generation APIs
