# AI Chat Template

1. Follow the [Interactive Workflow](https://vercel.com/new/clone?demo-title=Next.js+Chat&demo-description=A+full-featured%2C+hackable+Next.js+AI+chatbot+built+by+Vercel+Labs&demo-url=https%3A%2F%2Fchat.vercel.ai%2F&demo-image=%2F%2Fimages.ctfassets.net%2Fe5382hct74si%2F4aVPvWuTmBvzM5cEdRdqeW%2F4234f9baf160f68ffb385a43c3527645%2FCleanShot_2023-06-16_at_17.09.21.png&project-name=Next.js+Chat&repository-name=nextjs-chat&repository-url=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fai-chatbot&from=templates&skippable-integrations=1&env=OPENAI_API_KEY%2CAUTH_SECRET&envDescription=How+to+get+these+env+vars&envLink=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fai-chatbot%2Fblob%2Fmain%2F.env.example&teamCreateStatus=hidden&stores=%5B%7B%22type%22:%22kv%22%7D%5D) for creating a new project from this template, and all the necessary components

2. Finish the setup by following the instructions in the page and save the environment variables to the project

   - After the project is created, you will be redirected to the project dashboard
   - Click on "Visit" to navigate to the deployed website and interact with the AI chatbot
   - Click on "Repository" to navigate to the new repository in GitHub

3. Clone the [AI Chat Template](https://chat.vercel.ai/) from the [Github repository](https://github.com/vercel/ai-chatbot) to a folder in your local machine

   ```bash
    git clone https://github.com/<your-username>/<repository-name>.git
   ```

   - Replace `<your-username>` and `<repository-name>` with your GitHub username and the name of the repository you created in the previous step

4. Change the directory to the cloned project and install the dependencies

   ```bash
    cd nextjs-chat
    npm install
   ```

5. Link your local project to the Vercel project to get the environment variables from the settings and add them to the `.env` file in the root of the project

   ```bash
   vercel link
   ```

   - Choose **yes** to continue
   - Select the scope of the project (personal or organization)
   - Select the project to link

     - The CLI should detect the project just created in the previous steps

   - Pull the environment variables to the `.env` file

   ```bash
   vercel env pull
   ```

6. Start the development server

   ```bash
    npm run dev
   ```

7. Navigate to <http://localhost:3000/> to see the chat page in action
