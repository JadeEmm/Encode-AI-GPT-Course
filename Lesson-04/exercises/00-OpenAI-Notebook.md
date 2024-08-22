# Testing OpenAI API with a Colab Notebook

1. Navigate to the [Google Colab](https://colab.research.google.com/) website
2. Create a free account if you don't have one
3. Click on `+ New Notebook` in the bottom of the `Open Notebook` modal
   - If you closed the modal, click on `File` in the top of the screen and then click on `New notebook in Drive`
4. Give your notebook a name (for example, `OpenAI API Test`)
5. Click on the `+ Text` button to add a text cell
6. Add a title to the cell (for example, `OpenAI API Test`) by clicking on the `Toggle heading` button
7. Add a description to the cell (for example, `Testing OpenAI API with a Colab Notebook`)
8. Click on the `+ Code` button to add a code cell
9. Copy the code below and paste it in the first code cell of the notebook

   ```python
   !pip install openai
   ```

10. Copy the code below and paste it in a second code cell of the notebook

    ```python
    from openai import OpenAI
    client = OpenAI(api_key="PLACEHOLDER")
    ```

11. Log in to [OpenAI](https://platform.openai.com/)
12. Go to [API Keys](https://platform.openai.com/api-keys)
13. Click on `Create new secret key`
14. Name your key
15. Set permissions to `All`
16. Click on `Create secret key`
17. Copy the key and paste it in the notebook code cell to replace the placeholder where it says `PLACEHOLDER` like this:

    ```python
    from openai import OpenAI
    client = OpenAI(api_key="sk-...")
    ```

18. Copy the code below and paste it in a third code cell of the notebook

    ```python
    messages = [
     {
          "role": "user",
          "content": "What is the capital of France?"
     }
    ]
    model = "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    response.choices[0].message.content
    ```

> For more instructions on how to complete this in different Operational Systems, go to <https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key>
