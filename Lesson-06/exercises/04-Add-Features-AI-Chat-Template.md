# Modifying the AI Chat Template

1. Open the `signup-form.tsx` file under the `components` folder

2. Modify the `SignupForm` component to include a short description about the chat app

   - Locate the `SignupForm` component

   ```tsx
   export default function SignupForm() {
   const router = useRouter()
   const [result, dispatch] = useFormState(signup, undefined)

   useEffect(() => {
     if (result) {
       if (result.type === 'error') {
         toast.error(getMessageFromCode(result.resultCode))
       } else {
         toast.success(getMessageFromCode(result.resultCode))
         router.refresh()
       }
     }
   }, [result, router])

   return (
     <form
       action={dispatch}
       className="flex flex-col items-center gap-4 space-y-3"
     >
       <div className="w-full flex-1 rounded-lg border bg-white px-6 pb-4 pt-8 shadow-md md:w-96 dark:bg-zinc-950">
         <h1 className="mb-3 text-2xl font-bold">Sign up for an account!</h1>
         <div className="w-full">
   ```

   - Add a short description about the chat app

   ```tsx
   return (
     <form
       action={dispatch}
       className="flex flex-col items-center gap-4 space-y-3"
     >
       <div className="w-full flex-1 rounded-lg border bg-white px-6 pb-4 pt-8 shadow-md md:w-96 dark:bg-zinc-950">
         <h1 className="mb-3 text-2xl font-bold">Sign up for an account!</h1>
         <p className="mb-4 text-sm text-zinc-600 dark:text-zinc-400">
           Sign up for an account to start chatting with the AI!
         </p>
         <div className="w-full">
   ```

3. Modify also the `SignupForm` to include some content there too

   - Add a new input field for the user's invite code

   ```tsx
   <div className="mt-4">
     <label
       className="mb-3 mt-5 block text-xs font-medium text-zinc-400"
       htmlFor="invitationCode"
     >
       Invitation Code
     </label>
     <div className="relative">
       <input
         className="peer block w-full rounded-md border bg-zinc-50 px-2 py-[9px] text-sm outline-none placeholder:text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950"
         id="invitationCode"
         type="text"
         name="invitationCode"
         placeholder="Enter your invitation code"
       />
     </div>
   </div>
   ```

4. Open the `footer.tsx` file under the `components` folder

5. Modify the `FooterText` component to include some information about yourself

   - Modify the text to include your name

   ```tsx
   export function FooterText({
     className,
     ...props
   }: React.ComponentProps<"p">) {
     return (
       <p
         className={cn(
           "px-2 text-center text-xs leading-normal text-muted-foreground",
           className
         )}
         {...props}
       >
         Example AI chat application built with passion by [Your Name]. Powered
         by <ExternalLink href="https://nextjs.org">Next.js</ExternalLink> and{" "}
         <ExternalLink href="https://github.com/vercel/ai">
           Vercel AI SDK
         </ExternalLink>
         . Connect with me on <ExternalLink href="https://linkedin.com/in/yourprofile">
           LinkedIn
         </ExternalLink>.
       </p>
     );
   }
   ```

6. Open the `header.tsx` file under the `components` folder

7. Modify the `Header` component to include a title for the chat app

   - Locate the `Header` component

   ```tsx
   export function Header() {
     return (
       <header className="sticky top-0 z-50 flex items-center justify-between w-full h-16 px-4 border-b shrink-0 bg-gradient-to-b from-background/10 via-background/50 to-background/80 backdrop-blur-xl">
         <div className="flex items-center">
           <React.Suspense fallback={<div className="flex-1 overflow-auto" />}>
             <UserOrLogin />
           </React.Suspense>
         </div>
         <div className="flex items-center justify-end space-x-2">
           <a
             target="_blank"
             href="https://github.com/vercel/nextjs-ai-chatbot/"
             rel="noopener noreferrer"
             className={cn(buttonVariants({ variant: "outline" }))}
           >
             <IconGitHub />
             <span className="hidden ml-2 md:flex">GitHub</span>
           </a>
           <a
             href="https://vercel.com/templates/Next.js/nextjs-ai-chatbot"
             target="_blank"
             className={cn(buttonVariants())}
           >
             <IconVercel className="mr-2" />
             <span className="hidden sm:block">Deploy to Vercel</span>
             <span className="sm:hidden">Deploy</span>
           </a>
         </div>
       </header>
     );
   }
   ```

   - Modify the title to include the name of the chat app

   ```tsx
   export function Header() {
   return (
    <header className="sticky top-0 z-50 flex items-center justify-between w-full h-16 px-4 border-b shrink-0 bg-gradient-to-b from-background/10 via-background/50 to-background/80 backdrop-blur-xl">
      <div className="flex items-center">
        <React.Suspense fallback={<div className="flex-1 overflow-auto" />}>
          <UserOrLogin />
        </React.Suspense>
      </div>
      <div className="flex items-center justify-end space-x-2">
        <h1 className="text-xl font-bold">Example AI Chat Application</h1>
      </div>
    </header>
   )

   ```

8. Remove the Github and Deploy buttons

9. Open the `chat-panel.tsx` file under the `components` folder

10. Modify the `exampleMessages` array to include some example messages that you prefer

    - Locate the `exampleMessages` array

    ```tsx
    const exampleMessages = [
      {
        heading: "What are the",
        subheading: "trending memecoins today?",
        message: `What are the trending memecoins today?`,
      },
      {
        heading: "What is the price of",
        subheading: "$DOGE right now?",
        message: "What is the price of $DOGE right now?",
      },
      {
        heading: "I would like to buy",
        subheading: "42 $DOGE",
        message: `I would like to buy 42 $DOGE`,
      },
      {
        heading: "What are some",
        subheading: `recent events about $DOGE?`,
        message: `What are some recent events about $DOGE?`,
      },
    ];
    ```

    - Modify the messages to include some of your own

    ```tsx
    const exampleMessages = [
      {
        heading: "Explain the concept of",
        subheading: "artificial intelligence",
        message: `Explain the concept of artificial intelligence in simple terms.`,
      },
      {
        heading: "What are the benefits of",
        subheading: "meditation?",
        message: "What are the benefits of practicing meditation regularly?",
      },
      {
        heading: "How can I improve my",
        subheading: "public speaking skills?",
        message: `How can I improve my public speaking skills and overcome stage fright?`,
      },
      {
        heading: "Describe the process of",
        subheading: `photosynthesis`,
        message: `Describe the process of photosynthesis in plants.`,
      },
    ];
    ```

11. Save the changes and run the project to see the modifications

12. Commit the changes to your project to trigger a new build on Vercel

13. Navigate to <localhost:3000> and see the changes live

14. If you want to update the live deployment, you can do so by pushing the changes to the main branch

    - Vercel will automatically detect the changes and deploy the new version

15. Check your deployment link to see the changes live
