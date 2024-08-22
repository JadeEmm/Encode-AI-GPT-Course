# Simple text to text curl example

Run this on your terminal:

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

You should see a response like this:

```json
{
  "id": "chatcmpl-9xJ3cbfpVXM97RxzV8beJc5bYzgSe",
  "object": "chat.completion",
  "created": 1723921636,
  "model": "gpt-4o-mini-2024-07-18",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 9,
    "total_tokens": 28
  },
  "system_fingerprint": "fp_507c9469a1"
}
```

For Windows users, use this command instead:

```cmd
curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer %OPENAI_API_KEY%" -d "{"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]}"
```

You can try different messages by changing the `content` field in the `messages` array.
Similarly, you can change the `model` field to use a different model. For example: `gpt-4o` or `gpt-4-0125-preview`.

> More info about OpenAI Chat API on <https://platform.openai.com/docs/api-reference/chat>
