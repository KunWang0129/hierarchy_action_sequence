"""
Lightweight local LLM client for inference

Usage:
    client = LocalLLMClient("llama3.1", "70b")
    response = client.generate(messages)
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = Path("/scratch/gpfs/ABDIENG/bw5889/models")
MODEL_REGISTRY: Dict[Tuple[str, str], str] = {
    # Llama models
    ("llama3.1", "70b"): "meta-llama/Llama-3.1-70B-Instruct",
    ("llama3.1", "8b"): "meta-llama/Llama-3.1-8B-Instruct",
    # Qwen models
    ("qwen3", "235b"): "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    ("qwen3", "32b"): "Qwen/Qwen3-32B",
    ("qwen3", "8b"): "Qwen/Qwen3-8B",
    ("qwen3", "4b"): "Qwen/Qwen3-4B",
    ("qwen3", "1.7b"): "Qwen/Qwen3-1.7B",
    ("qwen3", "0.6b"): "Qwen/Qwen3-0.6B",
    # DeepSeek R1 models
    ("deepseek-r1", "70b"): "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ("deepseek-r1", "32b"): "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # GPT-OSS models
    ("gpt-oss", "120b"): "openai/gpt-oss-120b",
    ("gpt-oss", "20b"): "openai/gpt-oss-20b",
}


def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
    try:
        return getattr(torch, dtype)
    except AttributeError:
        return None


class LocalLLMClient:
    def __init__(
        self,
        family: str,
        size: str,
        dtype: str = "auto",
        device_map: Optional[str] = "auto",
    ) -> None:
        key = (family.lower(), size.lower())
        local_subdir = MODEL_REGISTRY.get(key)
        if not local_subdir:
            raise ValueError(f"Unsupported local model: {family} {size}")

        model_path = BASE_PATH / local_subdir
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        torch_dtype = _resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    @staticmethod
    def available_models() -> Iterable[Tuple[str, str]]:
        return tuple(MODEL_REGISTRY.keys())

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts = []
        for msg in messages:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        if seed is not None:
            torch.manual_seed(seed)

        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {"content": content}

    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Alias for generate() to match StarMakingEnvText interface"""
        return self.generate(messages, max_new_tokens, temperature, top_p, seed)
