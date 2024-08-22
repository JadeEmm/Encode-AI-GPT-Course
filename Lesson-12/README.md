# Lesson 12: Building Our First Local AI Application

In this lesson, we'll explore creating our first local AI application. We'll learn how to integrate locally hosted AI services into our applications, enabling text generation tasks directly on our devices.

We'll conduct practical exercises using a local API in our NextJS application, exploring the text generation capabilities offered by GPT templates.

Additionally, we'll delve into the technical details of loading GPTs and the inference process. We'll experiment with various parameters to optimize performance and compatibility based on our hardware, and evaluate how different models behave under these configurations.

We'll examine various model types and their major differences in terms of precision, performance, and hardware requirements.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Node.js installed on your device
  - [Node.js](https://nodejs.org/en/download/)
- Proficiency with `npm` and `npx` commands
  - Documentation: [npm](https://docs.npmjs.com/)
  - Documentation: [npx](https://www.npmjs.com/package/npx)
- Understanding of `npm install` and managing the `node_modules` folder
  - Documentation: [npm install](https://docs.npmjs.com/cli/v10/commands/npm-install)
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for cloning repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Next.js installed on your device project
  - [Next.js installation](https://nextjs.org/docs/getting-started/installation)
- Familiarity with Next.js
  - Documentation: [Next.js](https://nextjs.org/docs)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) tool installed on your device
  - Follow the repository instructions for installation
  - Ensure you have the necessary dependencies and configurations on your device
- A compatible GPT model downloaded and placed in the correct `models` folder for Text Generation WebUI
- An account at [Hugging Face](https://huggingface.co/)

## Review of Lesson 11

- Text Generation WebUI Features
- Parameters and configurations
- Extensions
- Flags
- API

## Building Applications Using Local AI Services

In the previous lesson, we learned how to use the Text Generation WebUI application to run GPT models on our devices for text generation tasks.

We completed an exercise to call the local API from a Python script, enabling us to serve text generation tasks from a local service running the Text Generation WebUI application.

With this capability, we can now integrate these services into applications running in our local environment.

When deploying our applications to the cloud, we can use the same technique to serve text generation tasks from a cloud service, exposing the API server to the internet, authenticating our users with our own logic, and eliminating the need for OpenAI's API calls entirely.

## Using a Local API for Chat Completion Tasks

- Practical exercise
  - Exercise 1: [Replacing OpenAI's Chat Completion API with a local API](./exercises/00-Chat-Completion.md) in our `Chef GPT` NextJS application

## Exploring AI Possibilities with Local APIs

Now that we can easily create web applications using modern web development frameworks and have a local API to serve text generation tasks, we can combine these concepts to explore the potential of AI.

Since the number of tokens passed/generated from AI Tasks in OpenAI API calls directly affects billing, we typically avoid excessive experimentation. However, with a local environment for running these tasks at marginal costs (such as electricity and hardware), we can explore the capabilities of these models in depth.

## Testing Prompts

- Practical exercises
  - Exercise 2: [Testing prompt formats and techniques](./exercises/01-Testing-Prompts.md) for text generation tasks
  - Exercise 3: [Crafting a prompt to fact-check statements](./exercises/02-Fact-Checking.md)
  - Exercise 4: [Designing a prompt to generate stories](./exercises/03-Story-Generation.md)

## Text Generation Quality and Precision

- Strengths of GPTs
- Limitations of GPTs
- Optimizing LLM performance
- Utilizing LLMs for creative writing

## Creating an Application for Storytelling

- Practical exercise
  - Exercise 5: [Building a storytelling application](./exercises/04-Story-Telling.md) using NextJS and a local API for text generation tasks

## Weekend Project

To consolidate this week's knowledge, students should complete the following project:

1. Create a GitHub repository for your project
2. Add all group members as collaborators
3. Create a README.md file with your project description
4. Use the `story-telling-app` as a base repository or create a new application from scratch using NextJS
5. Implement a table of characters that users can create for the story
   - Users should be able to add, edit, and delete characters
   - Each new character should have a name, description, and personality
6. Customize the prompt to generate a story using user-created characters, if any
7. Implement a summary of each character's role in the story after the full text has been generated
8. Test different models for story generation and compare their outputs
   - Evaluate how well the models "remember" user-created characters
   - Experiment with different **context window** sizes across models to observe their impact on output
   - Test models of varying sizes and observe how this influences the output
     - Use models compatible with your device, focusing on the experiment rather than overall story quality
9. Submit your project via the submission form

> Locate your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>
> > If you can't find your group, contact the program manager through Discord or email

## Next Steps

- In-depth exploration of GPTs
- Models and hardware requirements
- "Understanding" GPT training
- Fine-tuning Models
