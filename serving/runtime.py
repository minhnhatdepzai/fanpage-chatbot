"""Nạp base model + adapter (LoRA/VeRA) bằng Transformers + PEFT và sinh câu trả lời.

Dùng chung cho model server (serving/server.py) và bộ đánh giá (chạy in-process).
Đường Transformers + PEFT hỗ trợ cả VeRA - nhiều inference engine tối ưu (vd. vLLM) chỉ hỗ trợ LoRA.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Tham số lấy mẫu khuyến nghị trong model card Qwen3-4B-Instruct-2507
DEFAULT_TOP_K = 20


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: int


class ModelRuntime:
    def __init__(
        self,
        base_model_id: str,
        revision: str,
        *,
        adapter_path: str | None = None,
        adapter_id: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_input_tokens: int = 6000,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.base_model_id = base_model_id
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.device = device
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=revision)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_id, revision=revision, dtype=getattr(torch, dtype), device_map=device, attn_implementation="sdpa"
        )
        self.base.eval()
        self.model: Any = self.base
        self.adapter_id: str | None = None
        self.adapter_method: str | None = None
        if adapter_path:
            self.load_adapter(adapter_path, adapter_id or Path(adapter_path).name)
        log.info("model_loaded", extra={"model": base_model_id, "seconds": round(time.perf_counter() - t0, 1)})

    # ------------------------------------------------------------------ adapter
    def load_adapter(self, path: str, adapter_id: str) -> None:
        from peft import PeftConfig, PeftModel

        with self._lock:
            cfg = PeftConfig.from_pretrained(path)
            if cfg.base_model_name_or_path and cfg.base_model_name_or_path != self.base_model_id:
                raise ValueError(
                    f"Adapter được huấn luyện trên {cfg.base_model_name_or_path}, không khớp base {self.base_model_id}"
                )
            self._unload_locked()
            self.model = PeftModel.from_pretrained(self.base, path, is_trainable=False)
            self.model.eval()
            self.adapter_id = adapter_id
            self.adapter_method = str(cfg.peft_type.value if hasattr(cfg.peft_type, "value") else cfg.peft_type)
        log.info("adapter_loaded", extra={"adapter_id": adapter_id, "method": self.adapter_method})

    def unload_adapter(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self.model is not self.base:
            self.base = self.model.unload()  # gỡ lớp adapter, trả lại base model nguyên vẹn
            self.model = self.base
        self.adapter_id = None
        self.adapter_method = None

    # ------------------------------------------------------------------ sinh
    def render(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # Transformers 5 returns BatchEncoding (a Mapping, not a plain dict).
        # Iterating it directly yields string keys such as "input_ids", which
        # later makes torch.tensor() fail with "too many dimensions 'str'".
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        return list(ids)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 400,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = DEFAULT_TOP_K,
        seed: int | None = None,
        stop: list[str] | None = None,
        use_adapter: bool = True,
    ) -> GenerationResult:
        torch = self.torch
        ids = self.render(messages)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"prompt quá dài: {len(ids)} > {self.max_input_tokens} token")
        # use_adapter=False: chạy base model cho lượt này (so sánh A/B mà không phải gỡ adapter/khởi động lại)
        bypass = self.model.disable_adapter() if (not use_adapter and self.model is not self.base) else nullcontext()
        with self._lock, torch.inference_mode(), bypass:
            if seed is not None:
                torch.manual_seed(seed)
            input_ids = torch.tensor([ids], device=self.model.device)
            t0 = time.perf_counter()
            do_sample = temperature > 0
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            latency = int((time.perf_counter() - t0) * 1000)
        new = out[0][len(ids) :].tolist()
        finish = "stop" if (new and new[-1] == self.tokenizer.eos_token_id) or len(new) < max_new_tokens else "length"
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                if s and s in text:
                    text, finish = text.split(s, 1)[0].rstrip(), "stop"
        return GenerationResult(text, len(ids), len(new), finish, latency)

    def vram_gb(self) -> float:
        if self.torch.cuda.is_available():
            return round(self.torch.cuda.memory_allocated() / 1e9, 2)
        return 0.0
