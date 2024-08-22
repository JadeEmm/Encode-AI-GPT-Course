# Building a Multi-Modal Chat Application

1. Create a sample project using the Vercel AI SDK

   ```bash
   npm create-next-app@latest multi-modal-chatbot
   ```

2. Change to the project directory

   ```bash
    cd multi-modal-chatbot
   ```

3. Install the dependencies

   ```bash
   npm install ai @ai-sdk/openai
   ```

4. Configure OpenAI API key to a local environment variable

   - Create a `.env.local` file in the root of the project

   - Add the OpenAI API key variable in the file by inserting this: `OPENAI_API_KEY=xxxxxxxxx`

   - Replace `xxxxxxxxx` with your OpenAI API key

5. Create a Route Handler at `app/api/chat/route.ts`:

   ```tsx
   import { openai } from "@ai-sdk/openai";
   import { convertToCoreMessages, streamText } from "ai";

   // Allow streaming responses up to 30 seconds
   export const maxDuration = 30;

   export async function POST(req: Request) {
     const { messages } = await req.json();

     const result = await streamText({
       model: openai("gpt-4o"),
       messages: convertToCoreMessages(messages),
     });

     return result.toDataStreamResponse();
   }
   ```

6. Open the `app/page.tsx` file

7. Add the chat component to the page:

   ```tsx
   "use client";

   import { useChat } from "ai/react";

   export default function Chat() {
     const { messages, input, handleInputChange, handleSubmit } = useChat();
     return (
       <div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
         {messages.map((m) => (
           <div key={m.id} className="whitespace-pre-wrap">
             {m.role === "user" ? "User: " : "AI: "}
             {m.content}
           </div>
         ))}

         <form
           onSubmit={handleSubmit}
           className="fixed bottom-0 w-full max-w-md mb-8 border border-gray-300 rounded shadow-xl"
         >
           <input
             className="w-full p-2"
             value={input}
             placeholder="Say something..."
             onChange={handleInputChange}
           />
         </form>
       </div>
     );
   }
   ```

8. Import and implement a `useState` and a `useRef` hook:

   ```tsx
   import { useState, useRef } from "react";

   export default function Chat() {
      const { messages, input, handleInputChange, handleSubmit } = useChat();

      const [files, setFiles] = useState<FileList | undefined>(undefined);
      const fileInputRef = useRef<HTMLInputElement>(null);

      ...
   ```

9. Add a `div` inside the message content to upload files:

   ```tsx
   <div>
     {m?.experimental_attachments
       ?.filter((attachment) => attachment?.contentType?.startsWith("image/"))
       .map((attachment, index) => (
         <img
           key={`${m.id}-${index}`}
           src={attachment.url}
           width={500}
           alt={attachment.name}
         />
       ))}
   </div>
   ```

10. Modify the `form` element to handle the file upload:

    ```tsx
    <form
      className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl space-y-2"
      onSubmit={(event) => {
        handleSubmit(event, {
          experimental_attachments: files,
        });

        setFiles(undefined);

        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }}
    >
      <input
        type="file"
        className=""
        onChange={(event) => {
          if (event.target.files) {
            setFiles(event.target.files);
          }
        }}
        multiple
        ref={fileInputRef}
      />
      <input
        className="w-full p-2"
        value={input}
        placeholder="Say something..."
        onChange={handleInputChange}
      />
    </form>
    ```

11. Run the project

    ```bash
    npm run dev
    ```

12. Open the browser and navigate to <http://localhost:3000>
