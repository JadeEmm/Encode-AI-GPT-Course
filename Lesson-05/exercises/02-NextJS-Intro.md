# Building a Web Page with NextJS

1. Create a new folder for your projects:

   ```bash
   mkdir my-projects
   cd my-projects
   ```

   > Pick a _safe_ location on your computer to store your projects. You can use the `Documents`, `Desktop`, or any other folder you prefer.

2. Create a new NextJS project using the following command:

   ```bash
   npx create-next-app my-next-app
   ```

   - You can give any name to your project by replacing `my-next-app` with your preferred name
   - Pick all the default options when prompted
     - ✔ Would you like to use TypeScript? … No / **Yes**
     - ✔ Would you like to use ESLint? … No / **Yes**
     - ✔ Would you like to use Tailwind CSS? … No / **Yes**
     - ✔ Would you like to use `src/` directory? … **No** / Yes
     - ✔ Would you like to use App Router? (recommended) … No / **Yes**
     - ✔ Would you like to customize the default import alias (@/\*)? … **No** / Yes

3. Navigate to the newly created project folder:

   ```bash
   cd my-next-app
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open your browser and navigate to `http://localhost:3000` to see your NextJS project running

6. Open the `app/page.tsx` file in your editor to make changes to the home page

7. Replace the existing code in `app/page.tsx` with the following:

   ```tsx
   export default function Home() {
     return (
       <>
         <h1>Hello World</h1>
         <p>This is a test</p>
       </>
     );
   }
   ```

8. Save the file and refresh your browser to see your changes

9. Apply some styling to the page using the `Tailwind CSS` classes

   ```tsx
   export default function Home() {
     return (
       <>
         <div className="flex flex-col items-center justify-center min-h-screen">
           <h1 className="text-4xl font-bold mb-4 text-blue-600">
             Hello World
           </h1>
           <p className="text-xl text-gray-500">This is a test</p>
         </div>
       </>
     );
   }
   ```

10. Create some page elements

    ```tsx
    export default function Home() {
      return (
        <>
          <div className="flex flex-col items-center justify-center min-h-screen">
            <div className="flex flex-col items-center justify-center">
              <h1 className="text-4xl font-bold mb-4 text-blue-600">
                Hello World
              </h1>
              <p className="text-xl text-gray-500">This is a test</p>
            </div>
            <div className="flex flex-row items-center justify-center py-4">
              <input
                type="text"
                placeholder="Enter your name"
                className="bg-gray-200 p-2 rounded-md mr-2"
              />
              <button className="bg-blue-600 text-white p-2 rounded-md">
                Submit
              </button>
            </div>
          </div>
        </>
      );
    }
    ```

11. Import some functionalities from the `react` library:

    ```tsx
    "use client";
    import { useState } from "react";
    ```

12. Create some dynamic content in the page:

    ```tsx
    export default function Home() {
      const [name, setName] = useState("");
      const [showGreeting, setShowGreeting] = useState(false);

      const handleSubmit = () => {
        setShowGreeting(true);
      };

      return (
        <>
          <div className="flex flex-col items-center justify-center min-h-screen">
            <div className="flex flex-col items-center justify-center">
              <h1 className="text-4xl font-bold mb-4 text-blue-600">
                Hello World
              </h1>
              <p className="text-xl text-gray-500">This is a test</p>
            </div>
            <div className="flex flex-row items-center justify-center py-4">
              <input
                type="text"
                placeholder="Enter your name"
                className="bg-gray-200 p-2 rounded-md mr-2 text-black"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <button
                className="bg-blue-600 text-white p-2 rounded-md"
                onClick={handleSubmit}
              >
                Submit
              </button>
            </div>
            {showGreeting && (
              <div className="mt-4 p-4 bg-gray-100 rounded-md w-full max-w-2xl">
                <p className="text-3xl font-bold text-black">Hello {name}</p>
                <textarea
                  className="w-full h-40 mt-4 p-2 text-lg text-black bg-white border border-gray-300 rounded-md resize-none"
                  placeholder="Enter your message here..."
                ></textarea>
              </div>
            )}
          </div>
        </>
      );
    }
    ```
