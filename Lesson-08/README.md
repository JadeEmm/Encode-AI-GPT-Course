# Lesson 08: Exploring AI Capabilities

In this lesson, we'll delve into creating a multimodal AI application. We'll learn how to integrate multimodal AI models using OpenAI APIs into our applications, enabling us to perform various generation tasks with a single model.

Subsequently, we'll conduct practical exercises using different OpenAI models, exploring their text generation capabilities.

We'll also analyze the quality and accuracy of text generation based on various prompt techniques, identifying the most effective ways to prompt GPT models and areas that require more attention. Finally, we'll create an example application using NextJS and Vercel AI SDK, consolidating the concepts learned throughout the weeks and preparing for more advanced AI projects in the future.

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
- Basic knowledge of JavaScript programming language syntax
  - [JavaScript official tutorial](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/javascript/)
- Basic knowledge of TypeScript programming language syntax
  - [TypeScript official tutorial](https://www.typescriptlang.org/docs/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/typescript/)
- An account at [OpenAI Platform](https://platform.openai.com/)
  - To run API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 07

- Image generation with DALL-E
- Audio generation with TTS models
- Code generation models
- Code copilots
- Generating frontend code with AI
- Implementing components from V0

## Multimodal AI Applications

- Multimodal AI models overview
  - A single model to process text, images, and audio
  - Latency considerations
  - Function calling capabilities
  - Output format options
- The [Gemini model](https://cloud.google.com/use-cases/multimodal-ai)
- The [Claude 3 model family](https://www.anthropic.com/news/claude-3-family)
- The [GPT-4 Vision model](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/)
- Building a [simple multimodal application](https://sdk.vercel.ai/docs/guides/multi-modal-chatbot)

- Practical exercise
  - Exercise 1: [Multi-Modal Chat App](./exercises/00-Multi-Modal-Chat-App.md) to interact with multimodal AI models

## Exploring AI Possibilities with Local APIs

Having learned how to create web applications using modern web development frameworks and implement various generation tasks via API calls, we can now combine these concepts to create applications that explore the expanding capabilities of AI.

## Testing Prompts

- Practical exercises
  - Exercise 2: [Testing prompt formats and techniques](./exercises/01-Testing-Prompts.md) for text generation tasks
  - Exercise 3: [Crafting a prompt to fact-check statements](./exercises/02-Fact-Checking.md)
  - Exercise 4: [Designing a prompt to generate stories](./exercises/03-Story-Generation.md)

## Text Generation Quality and Precision

- Strengths of GPT models
- Limitations of GPT models
- Optimizing performance and accuracy from LLMs
- Utilizing LLMs for creative writing

## Creating an Application for Storytelling

- Practical exercise
  - Exercise 5: [Building a storytelling application](./project/04-Story-Telling.md) using NextJS and a local API for text generation tasks

## Weekend Project

To consolidate the knowledge acquired this week, students should complete the following project:

1. Create a GitHub repository for your project
2. Add all members of your group as collaborators
3. Create a README.md file with a description of your project
4. Create a new application from scratch using NextJS
5. Develop a page for generating jokes using AI
6. Add a feature for users to customize the Joke Parameters

   - Choose which parameters you'd like to offer your users
   - For example, allow users to select:
     - A topic from a list of options (work, people, animals, food, television, etc.)
     - A tone for the joke (witty, sarcastic, silly, dark, goofy, etc.)
     - The type of joke (pun, knock-knock, story, etc.)
     - The "temperature" (how much randomness/creativity to add to the joke)
   - Consider how you'll construct the prompt for the AI model to adhere to these parameters

7. After configuring the parameters, users should click a button to generate the joke, and the generated response must be displayed on the user's screen within the same page
8. Add a feature for the AI to evaluate if the generated jokes are "funny", "appropriate", "offensive", or other criteria you deem important
9. Experiment with different prompts and system instructions to optimize generalist AI models for "subjective" classification tasks like humor, appropriateness, offensiveness, etc.
10. Submit your project via the submission form

> Find your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>
> > If you can't locate your group, please contact the program manager through Discord or email

## Next Steps

- Python environment setup
- Running AI models on your own device
- Models and hardware requirements
