# Deploying the Application to Vercel

1. If you don't have a Vercel account, you can create a free (_Hobby_) one [here](https://vercel.com/signup)
2. Install the [Vercel CLI](https://vercel.com/docs/cli) by running the following command in your terminal:

   ```bash
   npm install -g vercel
   ```

3. Add the environment variables to the Vercel environments

   ```bash
   vercel env add OPENAI_API_KEY
   ```

   - Paste your OpenAI API key as the value for the `OPENAI_API_KEY` variable when prompted
   - Select all environments by pressing **Space** for each of the options
   - Hit **Enter** to confirm

4. Run the Vercel CLI deploy command in your terminal in the root of your project's folder to link and deploy your project:

   ```bash
   vercel 
   ```

   - Send **yes** to continue
   - Connect your Vercel account to the CLI
   - Choose the scope of the project (personal or organization)
   - Choose **N** for linking the repository to an existing project
   - Pick any project name you like
   - Input the path to the source of your project (should be `./`)
   - Vercel CLI should detect the framework as **Next.js** and set up the correct settings
   - Choose **N** for modifying the project's settings

5. Use the `--prod` flag to deploy the project to production for public usage

   - Projects deployed to _preview_ environments are not publicly accessible unless you change the project's [Deployment Protection Settings](https://vercel.com/docs/security/deployment-protection)

6. Follow the instructions in the terminal to deploy your project to Vercel

7. Pick the default options for NextJS deployment

8. Do not override the default settings

9. Once the deployment is complete, you will receive a URL to access your project

10. You may also use the Vercel dashboard to manage your project and your deployments

    - If you face errors due to the API keys, check the [Environment Variables](https://vercel.com/docs/deployments/environments) section of the project settings
