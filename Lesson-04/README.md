# Lesson 04: Using OpenAI APIs

In this lesson, we will explore how to use the OpenAI APIs to run GPTs for text-to-text tasks by invoking remote execution on a server that runs the GPTs for us.

For this lesson, you should have a basic understanding of how to use developer tools on your device. You should also have at least a fundamental grasp of Python programming language syntax. It's necessary to set up an account on the OpenAI Platform and have a minimum of 5 USD credits on your account to run the API commands in the OpenAI exercise.

We will experiment with OpenAI APIs to run Generative AI tasks on a remote server using Python scripts with the `openai` package.

We'll interact with various endpoints of the OpenAI API, such as `chat`, `image`, and `vision`, to test the capabilities of the Generative AI models available on the platform. We'll compare different models for these tasks and observe how they perform in various scenarios.

In future lessons, we'll run some of these models on our own devices, but for now, we'll use the OpenAI platform to simplify the learning process.

We'll delve deeper into the topic of Text Generation by studying how different `prompt` techniques can be used to control and fine-tune the output of Generative AI models.

## Prerequisites

- Familiarity with using a shell/terminal/console/bash on your device
  - Proficiency in basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Installation of Python tools on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency in using `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Knowledge of using `venv` to create and manage virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Installation of `git` CLI on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency in using `git` commands to clone repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Fundamental understanding of `Python` programming language syntax
  - [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
- Creation of an account at [OpenAI Platform](https://platform.openai.com/)
  - To run the API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 03

- Definition and functionality of GPTs
- GPT operational mechanisms
- Capabilities and limitations
- Use cases
- Applications utilizing GPTs
- Utilization of GPTs with the `transformers` package

## OpenAI APIs

- API Functionality
  - API calls enable remote execution of operations on servers over the internet, offering several advantages:
    - Leveraging substantial computational power not available on local devices
    - Processing large datasets that exceed local storage capabilities
    - Handling sensitive data (e.g., credentials, personal information) securely on remote servers
    - Protecting sensitive operations from client-side exposure
- Interacting with APIs over HTTP
  - APIs serve as essential interfaces for accessing and manipulating web service resources
  - In Python, the widely-used 'requests' library facilitates HTTP interactions:
    - Constructing HTTP requests:
      - Specify request method (GET, POST, PUT, DELETE, etc.)
      - Define API endpoint URL
      - Example: `response = requests.get('https://api.example.com/data')`
    - Processing API responses:
      - Retrieve data from response body
      - Access metadata (HTTP status codes, headers)
      - Example: `print(response.text)`
    - Implementing authentication:
      - Include credentials in request headers
      - Example:

        ```python
        headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
        response = requests.get('https://api.example.com/data', headers=headers)
        ```

    - Handling errors:
      - Check response status codes
      - Implement appropriate error handling
      - Example:

        ```python
        if response.status_code == 200:
            print('Request was successful')
        else:
            print('Error:', response.status_code)
        ```

  - Benefits of API-based processing:
    - Execution of resource-intensive tasks on powerful remote servers
    - Secure handling of sensitive data without local exposure
    - Efficient transfer of processed results to the client

This approach enables developers to leverage remote computational resources and data securely, while maintaining a lightweight client-side application.

- OpenAI API
  - The OpenAI API provides access to a variety of powerful artificial intelligence models trained on large datasets, including text generation, image generation, and natural language recognition models. Developers can integrate AI capabilities into their applications, websites, and systems by accessing models through specific API endpoints.
  - Since the use of this computation is billed, OpenAI needs to identify and authorize (or deny) each person/agent making requests to the API
    - This is achieved by assigning a unique **secret key** or set of keys, which are specific to each user and must be included in every request made to the API
  - Authentication to the OpenAI API is performed through the use of an **API key**. Each user is assigned a unique API key that must be included in every request made to the API. The API key is typically passed as an **HTTP header** in the request or as a query parameter.
  - Each request made to the API incurs costs; that is, each computation performed on the OpenAI API is charged to the user
  - Costs may vary based on the type of model used, the size of the request and response, and the amount of compute power and resources utilized
    - The models have different costs depending on the amount of data (tokens, pixels, etc.) they process
    - The [Pricing](https://openai.com/api/pricing/) page has the most up-to-date information on the costs of the different models and tasks
- Endpoints
  - Each endpoint available in the OpenAI API offers specific functionality to meet developers' needs
  - There isn't a single endpoint that can be used to access all the models and capabilities of the OpenAI services
    - Instead, there are several endpoints, each providing access to a specific model or set of models, and each with its own set of capabilities and requirements
  - The most common endpoints are **chat** for text generation tasks, **audio** for speech recognition and creation, **images** for image generation and helpers for passing images to chat completion endpoints, and **fine_tuning** for creating and managing fine-tuning jobs and models
    - There are many other useful endpoints for tasks such as creating assistants, moderating text content, calculating text embeddings, and more
    - For a comprehensive list of features, refer to the "Capabilities" section of the [OpenAI Platform Documentation](https://platform.openai.com/docs/overview) and the "Endpoints" section of the [API Reference](https://platform.openai.com/docs/api-reference/introduction)
- Text Generation
  - The **chat** endpoint in the OpenAI API allows users to request text generation based on the provided input, which is useful for text autocompletion, content generation, and language translation
  - [Chat Completions](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) tasks require passing at least a **model definition** and a **message** in the **request body**
    - In the body, you can pass many other parameters such as model configurations for **temperature** and **max_tokens**, and request options for **stream** and **user**
  - The response object will contain the generated text inside an object called **choices**, as well as additional metadata such as the **model** used, the **system_fingerprint** of the GPT version, and the **usage** statistics of the API request
- Image Generation
  - The **image** endpoint in the OpenAI API allows users to request the generation of images based on a given input. Users can provide a description or visual input, and the API generates corresponding images using specific artificial intelligence models designed for this purpose.
- Vision
  - OpenAI's **chat** endpoint allows passing one or multiple images as content in the request body, under the **content** object, by giving it a type of **image_url**
    - The provided images can then be used when the model is generating a response by first being processed through an _image recognition_ process, and then being used as _context_ for the _text generation_ process
      - According to the [documentation](https://platform.openai.com/docs/guides/vision), the model excels at answering general questions about the contents of images, but it's not yet optimized to answer detailed questions about the location of specific objects in an image
        - For example, you can ask about the color of a car or suggest dinner ideas based on the contents of your fridge, but if you show it an image of a room and ask where the chair is, it may not answer the question accurately
  - You can control the **detail** parameter to specify how the model processes the image and generates its textual understanding
    - You can provide three options: **low**, **high**, or **auto**
    - By default, the model will use the **auto** setting, which will examine the image input size and decide whether to use the **low** or **high** setting
      - Using **low** will enable the "low res" mode, where the model receives a low-resolution 512px x 512px version of the image and represents the image with a budget of 65 tokens
        - This allows the API to return faster responses and consume fewer input tokens for use cases that don't require high detail
      - Using **high** will enable "high res" mode, which first allows the model to see the low-res image and then creates detailed crops of input images as 512px squares based on the input image size
        - Each of the detailed crops uses twice the token budget (65 tokens) for a total of 129 tokens

## Implementing OpenAI API Calls

- The [OpenAI Python API library](https://github.com/openai/openai-python)
- Environment setup
- Authentication
- Calling the endpoints
- Handling the responses

## Using the OpenAI Python API library

- Practical exercises
  - Exercise 1: Experiment with LLMs using OpenAI APIs with a new [example notebook](./exercises/00-OpenAI-Notebook.md)
  - Exercise 2: [Setting up your OpenAI API keys](./exercises/01-OpenAI-Key.md) on your device
  - Exercise 3: Executing a [Simple text-to-text curl example](./exercises/02-Text-To-Text-Curl-Example.md)
  - Exercise 4: Building a simple Python script for [experimenting with models for text completion tasks](./exercises/03-Text-Completion-Models.md) to evaluate their performance and capabilities
    - Code and run [TextCompletionModelsTest.py](./examples/TextCompletionModelsTest.py), [TextCompletionModelsComparison.py](./examples/TextCompletionModelsComparison.py), and [TextCompletionModelsCutoff.py](./examples/TextCompletionModelsCutoff.py) in this exercise
  - Exercise 5: [Generating images with AI](./exercises/04-Image-Generation.md) to evaluate the performance and capabilities of the image generation models
    - Code and run [ImageGeneration.py](./examples/ImageGeneration.py) in this exercise
  - Exercise 6: [Using AI for computer vision tasks](./exercises/05-Computer-Vision.md) to evaluate the performance and capabilities of the computer vision models
    - Code and run [ComputerVision.py](./examples/ComputerVision.py) in this exercise

## Introduction to Prompt Engineering

- Importance of prompts
  - As we've seen in previous lessons, the prompt is a crucial element in controlling the output of Generative AI models
  - For each desired **output**, there is a set of corresponding **prompts** that can be passed to the model to get results in that **text region**
    - There's no direct way to know the exact **prompts** that will generate the desired **output**, especially when the criteria for "desired" are subjective and dynamic
    - The only possible way to _optimize_ the prompting process is to iteratively experiment with different **prompts** and observe the **outputs** to find the best combination of words in the **text space**
- Different prompts for different models
  - Each model has its own **text space functions** and **prompting** capabilities, so it's common to find prompts that work well for one model but not for another
  - Some models require specific structures and/or codes to be passed in the prompts to work properly
  - Some models are designed to be prompted with text to complete, others with image bytes, others with code snippets, others with step-by-step instructions, and so on
    - Due to this vast variety of requirements, it's almost impossible to find a one-size-fits-all prompt guide
- General recommendations for prompt techniques
  - **Specificity**: Clearly defining the question or task
  - **Contextual Clues**: Providing relevant background information
  - **Direct Instructions**: Explicitly stating what type of response is needed
  - **Tone Setting**: Indicating the desired style or tone of the response
- Other techniques
  - **Phrasing Variations**: Experimenting with different ways of phrasing the same question to see which elicits the best response
  - **Example Providing**: Giving examples of the expected output
  - **Guided Queries**: Asking follow-up questions to narrow down the focus
  - **Role-playing**: Asking the AI to assume a specific role or perspective to tailor its responses
  - **Constraints Setting**: Specifying any limitations or constraints for the response, such as word count, format, or specific points to include or avoid
  - **Iterative Refinement**: Revising and refining the prompt based on the AI's initial response to steer it in the desired direction

## Testing Prompts

- Practical exercises
  - Exercise 7: [Testing prompts for text completion tasks](./exercises/06-Testing-Prompts.md) to evaluate the performance and capabilities of different prompts
    - Code and run [TestingPrompts.py](./examples/TestingPrompts.py) in this exercise
  - Exercise 8: Building [Chef GPT script](./exercises/07-Chef-GPT.md) to suggest recipes and cooking tips based on a pre-configured prompt
    - Code and run [ChefGPT.py](./examples/ChefGPT.py) in this exercise

## Weekend Project

To consolidate the knowledge acquired this week, students should complete the following project:

1. Create a new GitHub repository for your project.
2. Invite all members of your group to collaborate on the repository.
3. Write a simple README.md file explaining your project
4. Modify and expand the `Chef GPT script` by incorporating a unique personality for your AI chef
   - Tweak the system prompt to include a unique personality for your AI chef
   - Example personalities:
     - A young, enthusiastic Indian chef specializing in Biryani
     - A seasoned Italian chef with a passion for pasta-making
     - An old Brazilian grandma who loves to cook classic dishes
5. Develop individual scripts for each group member, each featuring a distinct AI chef personality
6. Program the AI to respond to three specific types of user inputs:
   a. Ingredient-based dish suggestions
   b. Recipe requests for specific dishes
   c. Recipe critiques and improvement suggestions
7. Give enough instructions in the system prompt to make the AI conform to give the responses according to the scenarios above
   - Implement the following logic:
   - If the user's initial input doesn't match these scenarios, politely decline and prompt for a valid request.
     - For ingredient inputs: Suggest only dish names without full recipes.
     - For dish name inputs: Provide a detailed recipe.
     - For recipe inputs: Offer a constructive critique with suggested improvements.
   - Ideally, the same AI would be able to handle the three scenarios above
     - But if you can't get the same AI to do that, you can make three different scripts for each personality to proceed with your project
8. Conduct a comprehensive experiment:
   - The first person should use one script with one personality to suggest a dish based on given ingredients
     - After running the script, send the response for one of your group members (via Discord or any other means)
   - The second person should request a recipe for that dish using a second script with a different personality
     - After running the script, send the response for another of your group members
   - The third person should critique the provided recipe using a third script with a different personality
9. Compile a simple report documenting:
   - The experiment process
   - The system prompts used in each script
   - Comparative analysis of the different user prompts and their responses
10. Submit your completed project through the designated submission form.

> You should find your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>
> > If you can't find your group, please contact the program manager through Discord or email

## Next Steps

- Exploring other API Calls
- Exploring other providers
- Creating Frontend applications for your AI tasks
- Running models on your own hardware
