# Building a Chat Application with OpenAI Assistants

1. Clone the [Vercel AI SDK Quickstart Template](https://vercel.com/templates/next.js/openai-assistants-quickstart) for Assistants

   ```bash
   git clone https://github.com/vercel/openai-assistants-quickstart.git
   cd openai-assistants-quickstart
   ```

2. Install the dependencies

   ```bash
   npm install
   ```

3. Setup an Assistant in the OpenAI Platform with File Search and Code Interpreter enabled

4. Set the environment variables

   ```bash
   cp .env.example .env
   code .env
   ```

   - Modify the `.env` file with your OpenAI API key and Vercel AI SDK UI credentials

   ```text
   OPENAI_API_KEY="sk_..."
   OPENAI_ASSISTANT_ID="123..."
   ```

5. Run the application

   ```bash
   npm run dev
   ```

6. Open the application in your browser at <http://localhost:3000>

7. Test the application

   - Basic Chat Example: <http://localhost:3000/examples/basic-chat>
   - Function Calling Example: <http://localhost:3000/examples/function-calling>
   - File Search Example: <http://localhost:3000/examples/file-search>
   - Full-featured Example: <http://localhost:3000/examples/all>
