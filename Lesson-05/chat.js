document.getElementById("generate").addEventListener("click", sendMessage);

function sendMessage() {
    const message = document.getElementById("message").value; // Get the message from the input field
    const url = "https://api-inference.huggingface.co/models/vennify/t5-base-grammar-correction"; // API endpoint
    const data = {
        inputs: message // Request body: the message is assigned to the 'inputs' key
    };

    const options = {
        method: "POST",
        body: JSON.stringify(data), // Convert the data object to a JSON string and send it
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer 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" // Replace with your actual API key
        }
    };

    fetch(url, options)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(data); // Log the response to the console for debugging

            // Extract the generated text from the response
            const generatedText = data.generated_text || (data[0] && data[0].generated_text);
            const chatMessages = document.getElementById("chatMessages");
            const botMessage = document.createElement('div');
            botMessage.textContent = "Bot: " + (generatedText || "No response");
            chatMessages.appendChild(botMessage);
        })
        .catch(error => {
            console.error("Error:", error);
        });
}