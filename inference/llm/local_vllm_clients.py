"""
Local vLLM client mirroring the HuggingFace-based LocalLLMClient interface.

Usage:
    client = LocalVLLMClient("llama3.1", "70b")
    response = client.generate(messages)
"""

from typing import Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .local_llm_clients import BASE_PATH, MODEL_REGISTRY


class LocalVLLMClient:
    def __init__(
        self,
        family: str,
        size: str,
        dtype: str = "auto",
        device_map: Optional[str] = "auto",
        tensor_parallel_size: int = 1,
    ) -> None:
        del device_map  # Parity with LocalLLMClient signature; vLLM handles placement internally.

        key = (family.lower(), size.lower())
        local_subdir = MODEL_REGISTRY.get(key)
        if not local_subdir:
            raise ValueError(f"Unsupported local model: {family} {size}")

        model_path = BASE_PATH / local_subdir
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = {
            "model": str(model_path),
            "tokenizer": str(model_path),
            "tensor_parallel_size": tensor_parallel_size,
        }
        if dtype != "auto":
            llm_kwargs["dtype"] = dtype

        self.llm = LLM(**llm_kwargs)

    @staticmethod
    def available_models() -> Iterable[Tuple[str, str]]:
        return tuple(MODEL_REGISTRY.keys())

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

        parts = []
        for msg in messages:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        prompt = self._apply_chat_template(messages)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            n=1,
            best_of=1,
            stop_token_ids=[self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else None,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        content = outputs[0].outputs[0].text.strip()
        return {"content": content}

    def generate_batch(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Process multiple conversations in batch."""
        prompts = [self._apply_chat_template(msgs) for msgs in message_batches]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            n=1,
            best_of=1,
            stop_token_ids=[self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else None,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [{"content": out.outputs[0].text.strip()} for out in outputs]
