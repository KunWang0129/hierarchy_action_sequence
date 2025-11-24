from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
from google import genai
from google.genai import types
from tqdm import tqdm


class BaseLLMClient(ABC):
    @abstractmethod
    async def create_message(self, messages: List[Dict[str, str]], with_tools: bool = False) -> Any:
        pass

    @abstractmethod
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        pass

    @abstractmethod
    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        """Process multiple message conversations in batch for efficiency"""
        pass




class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", thinking_budget: int = 256):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.thinking_budget = thinking_budget

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None) -> Any:
        # Convert messages to the format expected by the new genai client
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled via config in the new API
                contents.append(f"System: {content}")
            else:
                contents.append(content)
        
        # Combine all content parts
        combined_content = "\n".join(contents)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=combined_content,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                )
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {"name": tool.get("name", ""), "description": tool.get("description", "")}

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use.get("name", ""), tool_use.get("arguments", {})

    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        """
        Gemini does not support batch processing, so we process sequentially.
        """
        responses = []
        with tqdm(total=len(message_batches), desc="Processing Gemini requests (sequential)", unit="req") as pbar:
            for i, messages in enumerate(message_batches):
                try:
                    response = self.create_message(messages, kwargs=kwargs)
                    responses.append(response)
                except Exception as e:
                    print(f"Error processing batch item {i}: {e}")
                    responses.append(f"Error: {str(e)}")
                pbar.update(1)
        return responses

    def get_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024,
                     temperature: float = 0.7, top_p: float = 0.9, seed: int = None) -> Dict[str, str]:
        """Generate response matching LocalLLMClient interface."""
        response_text = self.create_message(messages)
        return {"content": response_text}

    def generate_batch(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = None,
    ) -> List[Dict[str, str]]:
        """Generate responses for multiple message batches (sequential for Gemini)."""
        results = []
        for messages in tqdm(message_batches, desc="Gemini batch", unit="req"):
            response_text = self.create_message(messages)
            results.append({"content": response_text})
        return results