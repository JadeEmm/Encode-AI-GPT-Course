# Lesson 16: Enhancing OpenAI Assistants with RAG

OpenAI Assistants can leverage the extensive capabilities of the OpenAI infrastructure for efficient processing of functions like File Search, while abstracting away the complexity of underlying algorithms. However, these benefits come at a cost, similar to the situations we studied when replacing cloud-based LLM inference with locally hosted LLM inference.

In this lesson, we will explore the fundamentals of RAG-based inference by utilizing the [SuperBIG](https://github.com/kaiokendev/superbig) extension for Text Generation WebUI. Subsequently, we will implement a comprehensive RAG pipeline within a chat application using [LlamaIndex](https://www.llamaindex.ai/) TypeScript libraries.

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

## Review of Lesson 15

- Retrieval Augmented Generation (RAG)
- Integrating OpenAI Assistants with RAG
- Developing a chat application using OpenAI Assistants
- Implementing RAG in the chat application
- Vercel SDK Quickstart Template for Assistants

## Retrieval-Augmented Generation (RAG) Tools

- Web search capabilities
- File content search
- Video content analysis
- PDF document and book reading
- Windows-specific tool: NVIDIA [Chat with RTX](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/)
- Python package example: [Haystack](https://haystack.deepset.ai/)
- Multi-language tool (Python and TypeScript): [LlamaIndex](https://www.llamaindex.ai/)
- Enterprise-grade RAG solution: [LangChain](https://www.langchain.com/retrieval)

## Implementing a RAG Pipeline

- **Key components of a RAG pipeline**:

  - Document loader: Imports documents from various sources (e.g., PDFs, web pages, databases)
  - Text splitter: Divides documents into manageable chunks for processing
  - Text embedding model: Converts text chunks into numerical vectors
  - Vector store: Stores and indexes embedded text chunks for efficient retrieval
  - Retriever: Identifies relevant text chunks based on a query
  - Language model: Generates responses using retrieved information and the query

- **Implementation steps**:

  1. Load and preprocess documents
  2. Split documents into chunks
  3. Create embeddings for text chunks
  4. Store embeddings in a vector database
  5. Implement the retrieval mechanism
  6. Integrate with a language model for generation

- **Practical considerations**:

  - Optimal chunk size and overlap
  - Embedding model selection
  - Vector store scalability
  - Retrieval strategies (e.g., similarity search, hybrid search)
  - Prompt engineering for effective utilization of retrieved context

- **Advanced RAG techniques**:
  - Multi-document RAG
  - Hierarchical RAG
  - Iterative refinement
  - Hybrid search (combining keyword and semantic search)

## Experimenting with RAG using Text Generation WebUI

- Text Generation WebUI [extensions](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions)
- SuperBIG [GitHub repository](https://github.com/kaiokendev/superbig)
  - The [superbooga](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superbooga) and [superboogav2](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superboogav2) extensions
- Utilizing RAG with Chat mode
- Applying RAG in Instruction mode
- Leveraging RAG in the Notebook tab
- Effective dataset and chunk management
  - Balancing accuracy and context size
  - Determining optimal chunk size and overlap
  - Selecting appropriate embedding models
  - Ensuring vector store scalability
  - Refining prompt engineering for effective context utilization
- Exploring API methods for superbooga
- Managing the ChromaDB database

## Hands-on RAG Experience

- Practical exercise:
  - Exercise 1: [Integrate a local dataset](./exercises/00-RAG-With-Local-Dataset.md) with the superbooga extension and test it using a message

## Constructing a RAG Pipeline with LlamaIndex

- LlamaIndex [Python package](https://docs.llamaindex.ai/en/stable/)
- LlamaIndex [TypeScript package](https://ts.llamaindex.ai/)
- The [TypeScript Playground](https://github.com/run-llama/ts-playground) example project
- Essential [concepts](https://ts.llamaindex.ai/getting_started/concepts)
- Integrating LlamaIndex into a Next.js application

- Practical exercise:
  - Exercise 2: Utilize the [LlamaIndex TypeScript Playground](https://github.com/run-llama/ts-playground) to develop a RAG pipeline for chunking uploaded PDF files and generating text based on their content

## Weekend Project

To consolidate this week's learning, complete the following project:

1. Create a GitHub repository for your project
2. Add all group members as collaborators
3. Create a README.md file with a comprehensive project description
4. Use the `rag-app` as a foundation or develop a new application from scratch using Next.js
5. Design a page with a single input field for PDF file uploads
   - Users should upload a book or similar content with characters and settings
6. Implement a button to **extract characters** from the uploaded file
7. Develop a RAG pipeline to extract characters from the uploaded file
   - Each character should have a name, description, and personality
8. Add a text area below the button to display the results
9. Convert the output into an array of objects and present it in a table format
10. Integrate the `rag-app` with the `story-telling-app` to enable users to create new stories using imported characters, reusing their descriptions and personalities
11. Submit your project through the designated submission form

> Locate your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>
> > If you cannot find your group, please contact the program manager via Discord or email

## Future Directions

- AI Agents
- Agentic RAG
- Exploring alternative generative AI models
- Computer Vision
