# Building the Chef GPT App

In this exercise, we'll transform our basic chat application into a virtual chef assistant. We'll enhance the user interface, add chef-specific functionality, and improve the AI's responses to cooking-related queries.

Follow these step-by-step instructions to create the Chef GPT app, building upon your NextJS project from the previous exercise:

1. Open the `page.tsx` file

2. Set the chat container's height to 100% of the screen and improve its layout:

   - Add `h-screen` to the outer div to make it full height
   - Use `flex-col` to allow the chat container to grow and fill available space
   - Add `overflow-hidden` to prevent scrolling on the main container

   ```tsx
   <div className="flex flex-col w-full h-screen max-w-md py-24 mx-auto stretch overflow-hidden">
   ```

3. Organize the components with a `div` element

   - Wrap the chat messages in a div container to manage overflow, height, and scrolling as messages are generated:

   ```tsx
    <div className="flex flex-col w-full h-screen max-w-md py-24 mx-auto stretch">
      <div className="overflow-auto w-full">
        {messages.map((m) => (
          <div key={m.id} className="whitespace-pre-wrap">
            {m.role === "user" ? "User: " : "AI: "}
            {m.content}
          </div>
        ))}
      </div>
   ```

4. Extract additional properties from the [useChat hook](https://sdk.vercel.ai/docs/api-reference/use-chat) provided by Vercel AI SDK

   - Update the destructuring of the `useChat()` hook to include `append` and `isLoading`:

   ```tsx
   const {
     messages,
     input,
     handleInputChange,
     handleSubmit,
     isLoading,
     append,
   } = useChat();
   ```

   - `append`: Allows adding new messages to the chat programmatically
   - `isLoading`: Indicates whether the AI is currently processing a response

5. Next, create a conditional statement based on the `isLoading` variable to control what is rendered in this component

   - We'll display a loading indicator while messages are being generated

   - Add this code inside the div containing the chat messages (created in step 3), after the `messages.map` function's closing bracket:

   ```tsx
   {
     isLoading && (
       <div className="flex justify-end pr-4">
         <span className="animate-pulse text-2xl">...</span>
       </div>
     );
   }
   ```

   - This code checks if `isLoading` is true and, if so, displays an animated ellipsis

   - Explanation:
     - The `&&` operator is used for conditional rendering in React
     - If `isLoading` is true, the element after `&&` is rendered
     - If `isLoading` is false, nothing is rendered (short-circuit evaluation)
     - The `animate-pulse` class provides a subtle pulsing animation
     - `text-2xl` increases the size of the ellipsis for better visibility

6. Next, add styles to give the messages a chat bubble appearance

   - Update the `messages` element with the following code:

   ```tsx
   {
     messages.map((m) => (
       <div
         key={m.id}
         className={`whitespace-pre-wrap ${
           m.role === "user"
             ? "bg-green-700 p-3 m-2 rounded-lg"
             : "bg-slate-700 p-3 m-2 rounded-lg"
         }`}
       >
         {m.role === "user" ? "User: " : "AI: "}
         {m.content}
       </div>
     ));
   }
   ```

   - This is a common way to conditionally apply classes to elements based on the value of a variable

     - In this case, we are applying different classes to the chat bubbles based on the role of the message (user or AI)

     - The classes are defined in the `className` attribute of the `div` element, and the class names are generated based on the role of the message

     - The `whitespace-pre-wrap` class is used to preserve the whitespace and line breaks in the message content

     - The `bg-green-700` and `bg-slate-700` classes are used to set the background color of the chat bubbles, applied to the user and AI messages, respectively

     - The `p-3` class is used to add padding to the chat bubbles

     - The `m-2` class is used to add margin to the chat bubbles

     - The `rounded-lg` class is used to round the corners of the chat bubbles

     - The `User:` and `AI:` texts are added to the chat bubbles to indicate the role of the message sender

     - The `content` of the message is displayed inside the chat bubbles

     - The `key` attribute is used to uniquely identify each chat bubble according to the message `id` received in the API response

7. Check if you have edited all the components correctly

   - If you followed the steps until now, the code should look like this:

   ```tsx
   "use client";

   import { useChat } from "ai/react";

   export default function Chat() {
     const {
       messages,
       input,
       isLoading,
       append,
       handleInputChange,
       handleSubmit,
     } = useChat();
     return (
       <div className="flex flex-col w-full h-screen max-w-md py-24 mx-auto stretch">
         <div className="overflow-auto w-full">
           {messages.map((m) => (
             <div
               key={m.id}
               className={`whitespace-pre-wrap ${
                 m.role === "user"
                   ? "bg-green-700 p-3 m-2 rounded-lg"
                   : "bg-slate-700 p-3 m-2 rounded-lg"
               }`}
             >
               {m.role === "user" ? "User: " : "AI: "}
               {m.content}
             </div>
           ))}
           {isLoading && (
             <div className="flex justify-end pr-4">
               <span className="animate-bounce">...</span>
             </div>
           )}
         </div>

         <form onSubmit={handleSubmit}>
           <input
             className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl text-black"
             value={input}
             placeholder="Say something..."
             onChange={handleInputChange}
           />
         </form>
       </div>
     );
   }
   ```

8. Implement automatic scrolling for the chat container

   - To enhance user experience, we'll add automatic scrolling to ensure new messages are always visible

   - First, import the necessary React hooks at the top of your file:

   - Import `useEffect` and `useRef` from the `react` package at the top of the code

     - These are two functions from React that allow us to, respectively, perform side effects in functional components and access the properties of a DOM element

     ```tsx
     import { useEffect, useRef } from "react";
     ```

   - Create a reference for the messages container after the `useChat` hook line:

   ```tsx
   const messagesContainerRef = useRef<HTMLDivElement>(null);
   ```

   - Create a `useEffect` hook to scroll the messages container to the bottom whenever new messages are added:

   ```tsx
   useEffect(() => {
     if (messagesContainerRef.current) {
       messagesContainerRef.current.scrollTop =
         messagesContainerRef.current.scrollHeight;
     }
   }, [messages]);
   ```

   - Attach the reference to the messages `<div>` created in step 3:

   ```tsx
    <div className="overflow-auto w-full" ref={messagesContainerRef}>
   ```

9. Customize the page to add a button to prompt the AI to return a random recipe

   - Create a container at the bottom of your page

   - Wrap the `form` element with a `div` and move some classes from the `input` to the `div` container:

   ```tsx
   <div className="fixed bottom-0 w-full max-w-md">
     <form onSubmit={handleSubmit} className="flex justify-center">
       <input
         className="w-[95%] p-2 mb-8 border border-gray-300 rounded shadow-xl text-black"
         value={input}
         placeholder="Say something..."
         onChange={handleInputChange}
       />
     </form>
   </div>
   ```

   - Add a button to request a random recipe. This button should be placed above the form, inside the container we just created.

   - Use the `append` function from the `useChat` hook from step 4

   - Call this function when the user clicks the button with the following code:

   ```tsx
   <div className="flex flex-col justify-center mb-2 items-center">
     <button
       className="bg-blue-500 p-2 text-white rounded shadow-xl"
       disabled={isLoading}
       onClick={() =>
         append({ role: "user", content: "Give me a random recipe" })
       }
     >
       Random Recipe
     </button>
     <p className="p-4">OR</p>
   </div>
   ```

   - Notice that the button is still available to be clicked even when the AI is processing a message

     - To prevent this, We can use the same `isLoading` value (similar to step 5) to disable the button and the input

   - You might notice that the chat box may extend to the bottom of the page, where the new elements were added

     - To reserve some space for the new elements, you can add a margin to the bottom of the chat container

     - Add `mb-8` to the `input` class

   - The Code at this point should look like this:

   ```tsx
   "use client";

   import { useChat } from "ai/react";
   import { useEffect, useRef } from "react";

   export default function Chat() {
     const {
       messages,
       input,
       handleInputChange,
       handleSubmit,
       isLoading,
       append,
     } = useChat();

     const messagesContainerRef = useRef<HTMLDivElement>(null);

     useEffect(() => {
       if (messagesContainerRef.current) {
         messagesContainerRef.current.scrollTop =
           messagesContainerRef.current.scrollHeight;
       }
     }, [messages]);

     return (
       <div className="flex flex-col w-full h-screen max-w-md py-24 mx-auto stretch">
         <div className="overflow-auto mb-8 w-full" ref={messagesContainerRef}>
           {messages.map((m) => (
             <div
               key={m.id}
               className={`whitespace-pre-wrap ${
                 m.role === "user"
                   ? "bg-green-700 p-3 m-2 rounded-lg"
                   : "bg-slate-700 p-3 m-2 rounded-lg"
               }`}
             >
               {m.role === "user" ? "User: " : "AI: "}
               {m.content}
             </div>
           ))}

           {isLoading && (
             <div className="flex justify-end pr-4">
               <span className="animate-bounce">...</span>
             </div>
           )}
         </div>
         <div className="fixed bottom-0 w-full max-w-md">
           <div className="flex flex-col justify-center mb-2 items-center">
             <button
               className="bg-blue-500 p-2 text-white rounded shadow-xl"
               disabled={isLoading}
               onClick={() =>
                 append({ role: "user", content: "Give me a random recipe" })
               }
             >
               Random Recipe
             </button>
           </div>
           <form onSubmit={handleSubmit} className="flex justify-center">
             <input
               className="w-[95%] p-2 mb-8 border border-gray-300 rounded shadow-xl text-black"
               disabled={isLoading}
               value={input}
               placeholder="Say something..."
               onChange={handleInputChange}
             />
           </form>
         </div>
       </div>
     );
   }
   ```

10. Set up the AI to act as a chef by using system role definitions in the messages sent to the AI

    - Modify the code that requests the chat completions from OpenAI API in the `route.ts` file inside the `src/app/api/chat` folder to include a `system` message before the thread of messages in the chat

    ```typescript
    messages: [
      {
        role: 'system',
        content: 'You are a professional chef. You provide detailed cooking instructions, tips, and advice on selecting the best ingredients.',
      },
      ...messages,
    ],
    ```

    > In this code, we added a system message that tells the AI to act as a professional chef, providing detailed cooking instructions, tips, and advice on selecting the best ingredients.
    >
    > > You can experiment with different prompts to see how effectively the AI responds to different requests

11. Test the app to evaluate how the AI responds to the user's requests

    - Run the app by executing the following command in the terminal:

    ```bash
    npm run dev
    ```

    - Open the app on your browser at <http://localhost:3000/> and click the 'Random Recipe' button

    - Observe the AI's response

    - Try sending other messages to the AI and observe its responses

    - If the AI's responses are not as expected, you can tweak the code to improve the AI's performance
