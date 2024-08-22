# Creating a Simple Chat Page

1. Open the project created in the previous exercise in a text editor

2. Edit your `page.tsx` file to include the following code:

   ```tsx
   "use client";

   import React from "react";

   const Home = () => {
     const message = "Hello World!";

     return (
       <main className="min-h-screen bg-gray-900 py-6 flex flex-col justify-center sm:py-12">
         <h1 className="text-4xl font-bold text-center text-gray-100 mb-8">
           Chat Page
         </h1>
         <section className="max-w-3xl mx-auto w-full">
           <div className="bg-gray-800 shadow-lg rounded px-8 pt-6 pb-8 mb-4">
             <p className="text-xl text-gray-300">{message}</p>
           </div>
         </section>
       </main>
     );
   };

   export default Home;
   ```

3. Save the file and navigate to `http://localhost:3000` in your browser to view the changes

4. Stop the server by pressing `Ctrl+C` or `Cmd+C` in your terminal

5. Install the `openai` [npm package](https://www.npmjs.com/package/openai) by running `npm install openai` in your terminal

6. Create a `.env` file in the root of your project and add your OpenAI API key

   ```text
   OPENAI_API_KEY=sk-...
   ```

7. Create a folder called `api` in the root of your project

8. Create a folder named `chat` inside the `api` folder

9. Create a file named `route.ts` inside the `chat` folder

10. Paste the following code into the `route.ts` file:

    ```typescript
    import OpenAI from "openai";
    import { NextResponse } from "next/server";

    const openai = new OpenAI();

    export const runtime = "edge";

    export async function POST(req: Request) {
      const { messages } = await req.json();

      const response = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages,
      });

      return NextResponse.json({
        content: response.choices[0].message.content,
      });
    }
    ```

11. Modify the UI with the chat components:

    - In VSCode, open the file 'page.tsx' inside the folder 'app'

    - Replace the existing code with the following:

    ```typescript
    "use client";
    import { useState } from "react";
    
    const Home = () => {
      const [message, setMessage] = useState("");
      const [response, setResponse] = useState("");
    
      const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ messages: [{ role: "user", content: message }] }),
        });
        const data = await res.json();
        setResponse(data.content);
        setMessage("");
      };
    
      return (
        <main className="min-h-screen bg-gray-900 py-6 flex flex-col justify-center sm:py-12">
          <h1 className="text-4xl font-bold text-center text-gray-100 mb-8">
            Chat Page
          </h1>
          <section className="max-w-3xl mx-auto w-full">
            <div className="bg-gray-800 shadow-lg rounded px-8 pt-6 pb-8 mb-4">
              {!response && (
                <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Enter your message"
                    className="px-3 py-2 bg-gray-700 text-white rounded"
                  />
                  <button
                    type="submit"
                    className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
                  >
                    Send
                  </button>
                </form>
              )}
              {response && (
                <div className="mt-4 p-3 bg-gray-700 text-white rounded">
                  <p>{response}</p>
                </div>
              )}
            </div>
          </section>
        </main>
      );
    };
    
    export default Home;
    ```

12. Run the application: Now that everything is set up, you can start your application

    - On your terminal, run the following command:

    `npm run dev`

    - If everything is okay, you can now access the application at <http://localhost:3000/>
