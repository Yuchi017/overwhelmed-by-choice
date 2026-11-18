import asyncio
from openai import AsyncOpenAI

class AsyncVLLMClient:
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 3600,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def create_chat_completion(
        self,
        model: str,
        test_img_paths: list[str],
        prompt: str,
        max_tokens: int = 2048,
    ):      
        # print(f"Creating chat completion for model: {model}, image: {test_img_paths}")
        # print(f"Prompt:\n {prompt}")
        
        content = []
        
        if test_img_paths:
            for img_path in test_img_paths:
                if img_path is not None:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": "file://" + img_path
                        }
                    })

        # text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,   # ⭐ 關鍵
            top_p=1.0,         # ⭐ 保險
        )
        # print("\n" + "="*50)
        # print(f"Prompt: {prompt}")
        # print(f"Response: {response.choices[0].message.content}")
        # print("="*50 + "\n")
        return response.choices[0].message.content
