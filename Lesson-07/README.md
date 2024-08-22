# Lesson 07: Enhancing AI Features

Building upon our knowledge of the Vercel NextJS AI SDK, we'll now explore additional AI capabilities to integrate into our pages.

In this lesson, we'll expand on the basic chat app developed in the _Chef GPT_ exercise, incorporating advanced AI features for Image and Audio generation using OpenAI API endpoints. We'll customize the chat app to invoke the appropriate API endpoints with the correct formats and display the generated content dynamically using custom-built components.

Subsequently, we'll investigate how AI tools can assist developers in generating code snippets or even complete applications, evaluating their effectiveness and accuracy.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands
- Node.js installed on your device
  - [Node.js](https://nodejs.org/en/download/)
- Proficiency with `npm` and `npx` commands
  - Documentation: [npm](https://docs.npmjs.com/)
  - Documentation: [npx](https://www.npmjs.com/package/npx)
- Understanding of `npm install` and management of the `node_modules` folder
  - Documentation: [npm install](https://docs.npmjs.com/cli/v10/commands/npm-install)
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for repository cloning
  - Documentation: [Git](https://git-scm.com/doc)
- Basic knowledge of JavaScript programming language syntax
  - [JavaScript official tutorial](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/javascript/)
- Basic knowledge of TypeScript programming language syntax
  - [TypeScript official tutorial](https://www.typescriptlang.org/docs/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/typescript/)
- A free account on [Vercel](https://vercel.com/) to use the [V0](https://v0.dev/) application

## Review of Lesson 06

- Utilizing SDKs
- Vercel AI SDK UI
- Deploying applications
- Leveraging samples and templates
- Implementing the AI Chat Template

## Enhancing Frontend Features

In previous lessons, we explored the Vercel NextJS AI SDK to create a simple web application that generates text remotely using ChatGPT model API calls.

Our last exercise utilized the AI Chat Template from Vercel Templates to construct a robust chat application with numerous "out-of-the-box" features. While this template can be invaluable for building additional features, its structure, tailored for a specific chat flow, may present challenges when adapting it for applications that deviate from this flow.

To continue adding AI features without being constrained by the AI Chat Template's structure, we'll revisit the "Chef GPT" app, which generated recipes based on user prompts.

Let's now delve deeper into the capabilities of the Vercel NextJS AI SDK and incorporate more AI features into our web application.

## Expanding the Chef GPT App

- React concepts
  - [Conditional rendering](https://react.dev/learn/conditional-rendering)
  - [Forms](https://react.dev/reference/react-dom/components/form)
  - [Inputs](https://react.dev/reference/react-dom/components/input)
  - [useState](https://react.dev/reference/react/useState) hook
  - [useEffect](https://react.dev/reference/react/useEffect) hook
- AI Models

  - Image generation with [DALL-E](https://openai.com/index/dall-e-2/)
  - Audio generation with [TTS](https://platform.openai.com/docs/models/tts) models

- Practical exercises
  - Exercise 1: [Generating images for the dishes](./exercises/00-Generating-Images.md) to explore OpenAI APIs for image generation
    - Modify the project from the previous lesson's final exercise to incorporate new features
  - Exercise 2: [Generating audio instructions for the recipes](./exercises/01-Generating-Audio.md) to explore OpenAI APIs for text-to-speech
    - Further enhance the project from the previous lesson's final exercise

## AI Code Generation

To expedite frontend code creation for our web application, we can leverage Generative AI Tools.

Tools like ChatGPT and Copilot can not only enhance developer efficiency but also generate substantial code segments that serve as starting points for software projects.

While the accuracy of these tools isn't perfect, and the generated code may require adjustments and improvements, it often provides a valuable foundation for developers to "understand" feature or functionality implementation.

Some models are specifically designed and/or optimized for code generation. Unlike the general-purpose LLMs we've been using, these models are trained on code repositories and can generate code snippets based on input prompts. Users still need to integrate these snippets to form complete code, but the generated content can serve as a solid starting point for projects or new features. Examples include [Codex](https://openai.com/index/openai-codex/) from OpenAI, [CodeT5](https://github.com/salesforce/CodeT5) from Salesforce, and [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) from Meta.

Other models/tools are designed for versatility, capable of generating code snippets, providing instructions, explaining code, suggesting fixes, proposing tests, adding documentation, and more. These models can detect code context and generate the most appropriate suggestions based on user requests. Examples include [Copilot](https://copilot.github.com/) (powered by the Codex model), [TabNine](https://www.tabnine.com/), and [Replit](https://replit.com/). These tools can generate code snippets based on your IDE context while coding.

Some tools are designed to automatically generate and assemble entire applications based on simple descriptions of desired layout and functionality. These can be particularly helpful for developers creating frontend code for web applications.

A standout tool for this purpose is [v0](https://v0.dev/) from Vercel. `v0` is a web tool that generates frontend code based on simple descriptions of desired layout and functionality.

Another valuable tool is [MakeReal](https://makereal.tldraw.com/) from TLDraw. `MakeReal` is a web tool that generates frontend code based on simple descriptions and rough sketches of desired layout and elements.

These tools can significantly assist developers in creating frontend code for web applications and can be utilized to generate code for this lesson's exercises.

> Be advised that the generated code may require significant adjustments and improvements to function properly. Never rely blindly on AI-generated code.

## Utilizing Vercel V0 to Generate Components

- Using [Vercel V0](https://v0.dev/)
- Creating an initial UI from a prompt
- Making iterative changes
- Modifying layout and components
- Adding new components with prompts
- [Integrating the generated code](https://v0.dev/docs#adding-v0-components) with existing projects

- Practical exercises
  - Exercise 3: [Generating a new component for the Chef GPT app](./exercises/02-Generating-Component.md) to explore Vercel V0 for frontend code generation
    - Create a new project on Vercel V0 and generate frontend code for a new Chef GPT app component
    - Integrate the generated code with the project from the previous lesson's final exercise

## Next Steps

- Implementing a multimodal AI chat application
- Exploring OpenAI models
- Mastering prompt techniques
- Weekend project
