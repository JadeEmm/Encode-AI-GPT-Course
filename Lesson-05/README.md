# Lesson 05: Building a Simple Web Application

In this lesson, we will explore the process of creating AI applications using web technologies and AI models. We'll begin by examining the concept of a web application and learning how to construct a basic web page.

Our journey starts with the fundamentals of setting up a web development environment and crafting a simple web page using HTML, CSS, and JavaScript. We'll experiment with common web development tasks and then investigate how frameworks can help us build more complex applications efficiently.

Subsequently, we'll delve into using the Next.js React framework to construct a more sophisticated web application. We'll discover how Next.js can assist in building efficient and scalable applications while abstracting away many of the complexities associated with web technologies.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands such as `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
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

## Review of Lesson 04

- Weekend project review
- OpenAI APIs
- Handling requests
- Streaming responses
- Model configurations
- Prompt engineering

## What is a Web Application?

- Definition and characteristics of Web Applications
- HTML, CSS, and JavaScript: The building blocks
- Opening a web page in a browser

## Building a Simple Web Page

- Practical exercises
  - Exercise 1: [Creating a simple index.html file](./exercises/00-Index-HTML-Example.md) to grasp the basics of an HTML file
  - Exercise 2: [Generating a simple page with AI](./exercises/01-Web-Page-With-AI.md) to explore how ChatGPT can assist in coding a simple web page
    - Enhance the contents of your [index.html](./examples/index.html) file using AI-generated code

## Setting Up a Web Development Environment

- Essential tools for Web Development
  - **Text Editor**: A tool for writing and editing code, such as [Visual Studio Code](https://code.visualstudio.com/), [Sublime Text](https://www.sublimetext.com/), or [Atom](https://atom-editor.cc/)
  - **Web Browser**: A software application for accessing information on the World Wide Web, such as [Google Chrome](https://www.google.com/chrome/), [Mozilla Firefox](https://www.mozilla.org/en-US/firefox/new/), or [Microsoft Edge](https://www.microsoft.com/en-us/edge)
  - **Terminal**: A command-line interface for interacting with the operating system, such as [Windows Command Prompt](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands), [Windows PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview), [macOS Terminal](https://support.apple.com/guide/terminal/welcome/mac), or [Linux Terminal](https://ubuntu.com/tutorials/command-line-for-beginners)
  - **Version Control System**: A tool for tracking code changes and collaborating with other developers, such as [Git](https://git-scm.com/)
  - **JavaScript Runtime**: An environment that allows you to run JavaScript code outside of a web browser, such as [Node.js](https://nodejs.org/en/download/)
  - **Node Package Manager**: A package manager for JavaScript that allows you to install and manage dependencies for your projects, such as [npm](https://docs.npmjs.com/)
    - Alternatives include [Yarn](https://yarnpkg.com/), [pnpm](https://pnpm.io/), [bun](https://bun.sh/), or other package managers
- Using Node.js
  - Utilizing a package manager
  - The package.json file
    - Installing dependencies
    - Running scripts
  - The `node_modules` folder
  - Folder structure
- Using Git
  - Cloning a repository
  - Committing changes
  - Pushing changes
  - Pulling changes
  - Branching and merging
- Creating a Web Application project
- Utilizing starter-kits and frameworks

## Using JavaScript Frameworks

- Definition of a JavaScript Framework
  - A JavaScript framework is a collection of pre-written code that provides structures, libraries, and design patterns to facilitate web application development
    - JavaScript frameworks simplify common web development tasks, such as DOM manipulation, page routing, state management, asynchronous API calls, and form validation
    - They provide an organized and standardized architecture for applications, enhancing maintainability and scalability
- Implementing Frameworks
  - To begin using a framework, configure the project environment and install dependencies
  - Define the application structure, routes, components, and business logic using the framework's APIs and conventions
  - Frameworks address various development challenges, including:
    - **Complexity Abstraction**: Enabling greater focus on application logic rather than low-level implementation details
    - **Standardization of Practices**: Promoting best practices and coding conventions, ensuring project consistency and maintainability
    - **Efficiency and Productivity**: Providing reusable components, libraries, and utilities, eliminating the need to reinvent common features and functionality
    - **Enhanced Developer Experience**: Offering features such as real-time module reloading, code splitting, and comprehensive bug reporting to streamline the development workflow and accelerate iteration cycles
    - **Scalability and Performance Optimization**: Incorporating features like server-side rendering, on-demand loading, and caching mechanisms to optimize application performance and resource usage
- Selecting an Appropriate Web Development Framework
  - Three widely recognized JavaScript frameworks in the web development industry:
    - [Next.js](https://nextjs.org/docs): A full-stack React framework that simplifies modern web application development by offering features such as hybrid rendering, dynamic routing, and static pre-rendering
      - React is the most popular JavaScript library for building user interfaces, and Next.js elevates it to a full-fledged framework by adding features like server-side rendering, static site generation, and API routes
    - [Vue.js](https://vuejs.org/guide/introduction): Known for its simplicity and ease of learning, Vue.js offers a component-based approach and has a rich collection of plugins and extensions that facilitate the development of scalable and responsive applications
      - While not as popular as React, Vue.js is generally simpler to start with and has a strong community and rapidly growing ecosystem of tools and libraries
    - [Angular](https://angular.io/docs): Used for building large-scale and enterprise web applications, Angular offers a robust and comprehensive framework that includes powerful features such as dependency injection, page routing, and state management
      - Although Angular's popularity has declined in recent years, it can offer an easier learning curve for developers familiar with modular architectures and TypeScript syntax
- CSS Frameworks
  - CSS frameworks are collections of pre-written CSS styles and components used to style web applications quickly and consistently
    - They provide predefined classes and utilities that can be applied to HTML elements to achieve a desired visual appearance
    - CSS frameworks streamline the styling process and ensure consistent design across the application
  - Popular CSS frameworks include:
    - [Bootstrap](https://getbootstrap.com/docs/5.1/getting-started/introduction/): A front-end framework providing CSS and JavaScript components for building responsive and mobile-first web applications
    - [Tailwind CSS](https://tailwindcss.com/docs): A utility-first CSS framework that streamlines styling by providing pre-defined utility classes for direct use in HTML markup
    - [Bulma](https://bulma.io/): A modern CSS framework offering a collection of responsive and customizable components for web applications
    - [Materialize](https://materializecss.com/): A front-end framework implementing Google's Material Design guidelines, providing CSS and JavaScript components for modern and responsive web applications
  - UI Component libraries can be added to these frameworks to provide more complex, ready-to-use components:
    - [Material-UI](https://mui.com/): A popular React component library implementing Google's Material Design guidelines
    - [Ant Design](https://ant.design/): A comprehensive React component library offering a wide range of components and design patterns for enterprise-level web applications
    - [Chakra UI](https://chakra-ui.com/): A simple, modular component library for React providing accessible and customizable components
    - [shadcn/ui](https://ui.shadcn.com/): A collection of components and utilities for building web applications using Tailwind CSS
    - [Daisy UI](https://daisyui.com/): A utility-first CSS framework providing pre-defined utility classes using Tailwind CSS
- Utilizing the React.js Library
  - [React.js](https://react.dev/) is an open-source library introducing component-based development, where the user interface is divided into reusable and independent components
    - Each component represents a specific part of the user interface and can contain its own logic and internal state
    - Its main characteristic is the declarative approach to creating user interfaces
    - [React.js](https://react.dev/learn) is considered a library (not a framework) due to its modular approach, allowing developers to selectively import components when building a project
- Implementing the Next.js React Framework
  - [Next.js](https://nextjs.org/docs) is a React framework simplifying modern web application development by offering features such as hybrid rendering, dynamic routing, and static pre-rendering
    - Its philosophy centers on efficiency and scalability, providing a robust framework for rapid construction and simplified deployment
  - [Tailwind](https://tailwindcss.com/docs) is a utility-first CSS framework streamlining the styling process by providing pre-defined utility classes for direct use in HTML markup
    - Tailwind CSS is designed to be highly customizable and extendable, allowing developers to create unique and responsive designs without writing custom CSS

## Coding Using Next.js

- Framework Structure
  - Recent versions of Next.js allow project creation with two major structures: [Pages router](https://nextjs.org/docs/pages/building-your-application/routing/pages-and-layouts) or [App router](https://nextjs.org/docs/app/building-your-application/routing/pages-and-layouts)
    - The Pages router is the more common structure, where each file in the `pages` directory corresponds to a route in the application
    - The App router is a more recent and slightly more complex structure, where routes are governed by a JavaScript API instead of being implicitly defined by the file system
- Utilizing TypeScript and TSX
  - [TypeScript](https://www.typescriptlang.org/docs/) is a superset of JavaScript that adds static typing, providing type checking and code completion features that help catch errors early in the development process
    - TypeScript is a powerful tool for building large-scale applications, enforcing strict type checking and providing a more robust development experience
  - [TSX](https://www.typescriptlang.org/docs/handbook/jsx.html) is a syntax extension for TypeScript allowing developers to write JSX (JavaScript XML) in TypeScript files
    - JSX is a syntax extension for JavaScript enabling developers to write HTML-like code directly in their JavaScript files
    - TSX can be used to create [React Components](https://react.dev/learn/typescript) that are processed and rendered dynamically when the application is compiled/bundled for deployment
- Creating Components
  - Components are the building blocks of a React application, representing reusable and independent parts of the user interface
    - Components can be functional or class-based, containing their own logic, state, and lifecycle methods
  - Components can be created using the `function` keyword or the `class` keyword
    - Functional components are simpler and more concise, while class-based components offer more features and flexibility
  - Components can receive data through props, passed from parent components
    - Props are read-only and cannot be modified by the component
  - Components can also have internal state, managed using the `useState` hook
    - State allows components to store and update data locally
- Using Hooks
  - [Hooks](https://react.dev/reference/react/hooks) are functions allowing functional components to use state and other React features
    - Hooks provide a way to reuse stateful logic across components, facilitating the sharing and management of stateful logic in a React application
  - Common hooks include:
    - `useState`: Allows components to manage local state
    - `useEffect`: Enables components to perform side effects, such as data fetching and DOM manipulation
    - `useContext`: Allows components to access context values
    - `useRef`: Enables components to create mutable references to DOM elements
  - There are [recommended rules](https://react.dev/reference/rules/rules-of-hooks) for using hooks in React applications to avoid bugs and ensure proper functionality
- Implementing Tailwind CSS
  - [Tailwind CSS](https://tailwindcss.com/docs) is a utility-first CSS framework streamlining the styling process by providing pre-defined utility classes for direct use in HTML markup
    - Tailwind CSS is designed to be highly customizable and extendable, allowing developers to create unique and responsive designs without writing custom CSS
  - Tailwind CSS classes can be [applied directly to HTML elements](https://tailwindcss.com/docs/utility-first) for styling
    - Classes are used to apply styles such as colors, fonts, spacing, and layout to elements
    - These classes can be [reused for many elements](https://tailwindcss.com/docs/reusing-styles) and combined to create complex layouts and designs
  - Tailwind CSS provides utility classes for [responsive design](https://tailwindcss.com/docs/responsive-design), enabling developers to create layouts that adapt to different screen sizes
    - Responsive classes can be used to apply different styles based on screen size, such as hiding elements on mobile devices or changing the layout on larger screens
  - You can [customize your styles](https://tailwindcss.com/docs/adding-custom-styles) in Tailwind CSS by editing the configuration file and adding custom utility classes
    - Customizing Tailwind CSS allows you to create a unique design system for your entire application and tailor the framework to your specific needs

## Building an Application Using Next.js

- Practical exercises
  - Exercise 3: [Creating a simple React application using Next.js](./exercises/02-NextJS-Intro.md) to familiarize yourself with Next.js basics
    - Code and run a simple project using Next.js
  - Exercise 4: Add features by creating a simple [Chat Page](./exercises/03-Chat-Page.md) for users to interact with an AI model using OpenAI APIs
    - Edit the project created in the previous exercise to incorporate new features

## Next Steps

- Adding more features
- Deploying the Application
- Using SDKs
- Vercel AI SDK UI
- Utilizing samples and templates
