"""Embedding tiếng Việt (AITeamVN/Vietnamese_Embedding, fine-tune từ BAAI/bge-m3): CLS pooling + chuẩn hóa L2,
đúng cấu hình Sentence Transformers của model (1_Pooling: pooling_mode_cls_token, 2_Normalize). Không chạy mã đi kèm repo.
"""

from __future__ import annotations

import threading
from typing import Any


class EmbeddingRuntime:
    def __init__(
        self, model_id: str, revision: str, *, device: str = "cuda", max_length: int = 512, batch_size: int = 16
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model: Any = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype).to(device).eval()
        self.device = device
        self.dim = int(self.model.config.hidden_size)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        torch = self.torch
        out: list[list[float]] = []
        n_tokens = 0
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                n_tokens += int(batch["attention_mask"].sum())
                cls = self.model(**batch).last_hidden_state[:, 0].float()
                out.extend(torch.nn.functional.normalize(cls, dim=-1).cpu().tolist())
        return out, n_tokens
