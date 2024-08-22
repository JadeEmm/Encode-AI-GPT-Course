# Generating text with LLMs

1. Install all of the required dependencies for this exercise

   - Run `pip install bitsandbytes optimum`

   > Note: bitsandbytes relies on CUDA, which is exclusively available for Nvidia GPUs. Please note that installation instructions for Windows and MacOS differ from the general instructions. For detailed guidelines on installing bitsandbytes on Windows and MacOS, refer to the official documentation [here](https://huggingface.co/docs/bitsandbytes/installation)

2. Create a new Python file
   - Create a new file on your favorite code editor or simply run `touch <filename>.py` on your terminal (Linux/MacOS) or `type nul > <filename>.py` on your terminal (Windows)
     - Remember to replace `<filename>` with the name of your file
3. Import the `AutoModelForCausalLM`, `BitsAndBytesConfig`, and `AutoTokenizer` modules from the `transformers` package

   - Add the following line to your file using your favorite code editor:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   ```

4. Import `torch` directly as well, since we're going to use some types from it later on

   ```python
   import torch
   ```

5. Create a new instance of the Tokenizer in a variable to process the text inputs and also to decode generated outputs later on

   - We're going to use the `AutoTokenizer` class to automatically download the tokenizer suitable for the model we're using

   ```python
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", padding_side="left")
   ```

6. Create a quantization configuration object to use with the model

   - We're going to use the `QuantizationConfig` class to configure the quantization of the model

   ```python
   quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
   ```

   - You might need to change some of these [configuration parameters](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig) to better fit your hardware and the model you're using

7. Create a new instance of the LLM model in a variable using a `pre-trained model call` function

   - We're going to use the `microsoft/Phi-3-mini-4k-instruct` model, that will be automatically downloaded from the Hugging Face's model hub
   - We're going to use [4-bit dynamic quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to make the model run faster and use less memory, while sacrificing a little bit of accuracy
   - You can run with the `device_map="auto"` option to use more of your GPU(s) if you have one or many available on your device

   ```python
   model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", quantization_config=quantization_config, device_map="auto", low_cpu_mem_usage=True)
   ```

8. Use the Tokenizer to encode the input text and store the result in a variable

   ```python
   input_text = "What is the best recipe for Pepperoni pizza?"
   model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
   ```

9. Generate text using the model and the tokenized input text

   ```python
   generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
   ```

10. Decode the generated text and print the result

    ```python
    result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
    print(result)
    ```

11. Run the file

    ```bash
    # Linux/MacOS
    python <filename>.py
    ```

    ```bash
    # Windows
    py <filename>.py
    ```

12. Expect a considerable load time if you're running this command for the first time, as the `microsoft/Phi-3-mini-4k-instruct` model will be downloaded from the Hugging Face's model hub

    - The model is almost 8GB in size, so make sure you have enough disk space available on your device and a stable internet connection
    - The model will be downloaded to the `~/.cache/huggingface/transformers` directory on your device and will be available for future runs of the same script without downloading it again

13. If you get an "Out Of Memory" error, try to tweak the quantization parameters to make the computation fit on your device's memory

    - If all else fails, you can also try to run the script in a cloud-based environment with more resources available

14. If your script ran correctly, you should get an output like this:

    ```text
    What is the best recipe for Pepperoni pizza?

    A: Ingredients:

    2 1/4 tsp active dry yeast

    1/2 tsp sugar

    1 1/2 cup warm water (110 degrees F)

    3 cups bread flour

    2 tbsp olive oil

    1 tsp salt

    1. In a small bowl, dissolve the yeast and sugar in warm water. Let stand until creamy, about 10 minutes.

    2. In the bowl of an electric mixer fitted with a dough hook, combine the flour and salt. Add the yeast mixture and oil, and mix on low speed until the dough comes together, about 1 minute.

    3. Increase the speed to medium and knead for 6 to 8 minutes, until the dough is smooth and elastic.
    ```

> Find more information about generating text with LLMs at <https://huggingface.co/docs/transformers/main/en/llm_tutorial#generation-with-llms>
