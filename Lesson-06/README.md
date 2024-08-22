# Lesson 06: Building AI Applications with NextJS and Vercel

In the previous lesson, we grasped the basics of frontend development. Now, we'll delve into building, styling, and deploying AI applications with NextJS and Vercel. We'll explore advanced UI/UX practices and utilize Tailwind CSS for efficient styling.

We'll also learn to leverage the Vercel NextJS AI SDK to accelerate development, taking advantage of its numerous "out-of-the-box" features.

Finally, we'll clone and run a sample project from Vercel, using it as a foundation to construct our own AI application.

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
- A free account on [Vercel](https://vercel.com/) for application deployment

## Review of Lesson 05

- Web application fundamentals
- HTML, CSS, and JavaScript
- Frameworks and libraries
- React and NextJS
- TypeScript and TSX syntax
- Components, functions, and hooks
- Tailwind CSS
- Building applications with NextJS and Tailwind CSS
- Integrating OpenAI text generation APIs

## Building an Application Using Vercel NextJS AI SDK

- Leveraging boilerplate code and SDKs
- The Vercel [NextJS AI SDK](https://sdk.vercel.ai/docs/introduction)

  - Getting started with the SDK
    - Installation process
    - Implementing React Server Components
    - Utilizing providers
    - Streaming chat responses from APIs
    - Configuring Server Actions
  - Following the [Quick Start Guide](https://sdk.vercel.ai/docs/getting-started/nextjs-app-router) for App Router
  - Exploring the Vercel [AI SDK UI](https://sdk.vercel.ai/docs/ai-sdk-ui/overview) library
    - [Chatbot](https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot) example

- Practical exercises
  - Exercise 1: [Creating a simple web application with AI](./exercises/00-Vercel-NextJS-AI-SDK.md) to explore Vercel AI SDK utilities for the [OpenAI Provider](https://sdk.vercel.ai/providers/legacy-providers/openai)
    - Develop and run a basic project using Vercel NextJS AI SDK
  - Exercise 2: Enhance functionality by creating a [Chef GPT Page](./exercises/01-Chef-GPT-Page.md) to explore NextJS capabilities
    - Modify the project from Exercise 1 to incorporate new features

## Deploying the Application to a Hosting Service

- Web application hosting

  - Post-development, applications need deployment to make them accessible online
    - Static websites (HTML, CSS, JS) can utilize services like [GitHub Pages](https://pages.github.com/), [Netlify](https://www.netlify.com/), or file servers like [S3](https://aws.amazon.com/s3/) and [IPFS](https://ipfs.io/)
    - Server-Side Rendering (SSR) or Serverless Functions require services like [Vercel](https://vercel.com/) for server-side processing and API endpoints
  - Numerous free and paid hosting services are available
    - Selection depends on application requirements, budget, expected traffic, needed features, and desired developer control
    - For educational purposes, we'll use Vercel, which offers a suitable free tier for most projects

- CI/CD and DevOps for web applications

  - Automating deployment for code changes
    - Continuous Integration/Continuous Deployment (CI/CD) tools like [GitHub Actions](https://docs.github.com/en/actions) and [Vercel CLI](https://vercel.com/docs/cli) streamline this process
    - These tools create pipelines for automatic building and deployment upon code pushes
  - Vercel's free plan enables direct deployment from public GitHub repositories
    - Simplifies deployment by connecting the repository to Vercel for automatic updates
    - Vercel CLI allows command-line deployments, useful for non-GitHub hosted applications
  - Environment variables require separate configuration on Vercel

    - Set in the [Environment Variables](https://vercel.com/docs/environment-variables) section of project settings
    - Used for API keys, URLs, and sensitive information that shouldn't be hardcoded

- Practical exercise
  - Exercise 3: Use [Vercel](https://vercel.com/) to [deploy the project](./exercises/02-Deploy-Project.md) and make it publicly accessible
    - Deploy the project from previous exercises and publish it

## Utilizing Sample Projects and Templates

- Leveraging existing resources
  - Starting from scratch vs. using templates or sample projects
    - Templates provide a foundation with basic structure and features implemented
    - Useful for learning, accelerating development, or building upon existing projects
  - Vercel's [template project list](https://vercel.com/templates)
    - Covers a wide range of use cases from simple landing pages to complex web applications
    - Valuable for learning, feature development, or project initialization
- Bootstrapping with `create-next-app`
  - [NextJS Boilerplate](https://vercel.com/templates/next.js/nextjs-boilerplate) initialization via `npx create-next-app`
    - Creates a new NextJS project with basic structure and features
    - Serves as a foundation for feature development or new projects
- [App Router Playground](https://app-router.vercel.app/): A showcase of `App Router` structure features
  - Provides working examples for implementation in projects
- Starter Projects
  - [Starter Projects](https://vercel.com/templates?type=starter) page offers diverse project bases
    - Covers various use cases from simple to complex applications
    - Designed for easy adaptation to specific project needs
  - Developers can create custom starter templates for future projects
- AI Templates

  - [AI Templates](https://vercel.com/templates?type=ai) page showcases AI model and API integration
    - Demonstrates Vercel AI and UI SDK usage in AI applications
    - Includes implementations from various AI providers
  - [AI Chat Template](https://chat.vercel.ai/): A comprehensive base for AI chat applications
    - Preconfigured with App Router, React Server Components, Suspense, and Server Actions
    - Supports multiple AI providers including OpenAI, Anthropic, Cohere, Hugging Face, and custom models
    - Tailwind CSS styling with `shadcn/ui` components
    - Serverless authentication via Vercel and [NextAuth.js](https://github.com/nextauthjs/next-auth)
    - Chat history, rate limiting, and session storage using [Vercel KV](https://vercel.com/storage/kv)

- Practical exercises
  - Exercise 4: Clone and run the [AI Chat Template](./exercises/03-Clone-AI-Chat-Template.md) to explore sample project usage
    - Clone and locally run the AI Chat Template project
  - Exercise 5: Enhance the [AI Chat Template](./exercises/04-Add-Features-AI-Chat-Template.md) to further explore NextJS capabilities
    - Modify the cloned project to add new features

## Next Steps

- Integrating image generation APIs
- Incorporating audio generation APIs
- Generating frontend code with AI
- Implementing components from V0
