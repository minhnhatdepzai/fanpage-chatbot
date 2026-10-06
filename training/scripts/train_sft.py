"""Huấn luyện adapter LoRA hoặc VeRA (PEFT) trên dữ liệu SFT, chỉ tính loss trên token của assistant.

    uv run python -m training.scripts.train_sft --config training/configs/lora-edu-ent-v1.yaml --smoke
    uv run python -m training.scripts.train_sft --config training/configs/lora-edu-ent-v1.yaml

Không chạy cùng lúc với model server (cùng dùng GPU). Không tự triển khai adapter: sau khi train phải chạy
evaluation.scripts.compare rồi ``botctl model register/attach-eval/promote``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.conversation.knowledge import load_knowledge
from app.conversation.profile import load_profile
from training.sft_data import build_splits

IGNORE = -100


def encode(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """input_ids + labels: chỉ token thuộc câu trả lời assistant (kể cả <|im_end|>) được tính loss."""

    def ids(msgs: list[dict[str, str]], gen: bool) -> list[int]:
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    full = ids(messages, False)
    labels = [IGNORE] * len(full)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = ids(messages[:i], True)
        end = ids(messages[: i + 1], False)
        if full[: len(start)] != start or full[: len(end)] != end:
            raise ValueError("chat template không cho tiền tố ổn định - không che loss chính xác được")
        labels[len(start) : len(end)] = full[len(start) : len(end)]
    return full, labels


def masking_report(tokenizer: Any, input_ids: list[int], labels: list[int]) -> dict[str, Any]:
    spans, cur = [], []
    for tok, lab in zip(input_ids, labels, strict=True):
        if lab != IGNORE:
            cur.append(tok)
        elif cur:
            spans.append(tokenizer.decode(cur))
            cur = []
    if cur:
        spans.append(tokenizer.decode(cur))
    return {"tokens": len(input_ids), "trained_tokens": sum(lab != IGNORE for lab in labels), "trained_spans": spans}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="2 bước, lưu + nạp lại adapter để kiểm tra pipeline")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, VeraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s = get_settings()
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    kb = load_knowledge(s.knowledge_dir)
    dataset_paths = [Path(p) for p in cfg.get("datasets", [cfg.get("dataset")]) if p]
    splits, manifest = build_splits(dataset_paths, kb, load_profile(s.fanpage_profile_path))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_revision"])

    data: dict[str, list[tuple[list[int], list[int]]]] = {}
    dropped: list[str] = []
    for split, convs in splits.items():
        data[split] = []
        for c in convs:
            ids, labels = encode(tok, c)
            if len(ids) > cfg["max_seq_len"]:
                dropped.append(f"{split}:{len(ids)}")  # bỏ hẳn, không cắt -> không mất câu trả lời mục tiêu
                continue
            if not any(lab != IGNORE for lab in labels):
                raise ValueError("ví dụ không có token assistant nào được tính loss")
            data[split].append((ids, labels))
    mask_check = masking_report(tok, *data["train"][0])
    lengths = [len(x[0]) for x in data["train"]]
    print(json.dumps({"counts": {k: len(v) for k, v in data.items()}, "dropped_too_long": dropped,
                      "max_len": max(lengths), "masking_example": mask_check}, ensure_ascii=False, indent=1))

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"], dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    method = cfg["method"]
    if method == "lora":
        pcfg = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    elif method == "vera":
        pcfg = VeraConfig(task_type="CAUSAL_LM", **cfg["vera"])
    else:
        raise ValueError(f"method không hỗ trợ: {method}")
    model = get_peft_model(model, pcfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable_params={trainable} total={total} ({100 * trainable / total:.4f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    steps_per_epoch = math.ceil(len(data["train"]) / cfg["grad_accum"])
    max_steps = 2 if args.smoke else min(cfg["max_steps"], steps_per_epoch * cfg["epochs"])
    warmup = max(1, int(cfg.get("warmup_ratio", 0.1) * max_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, max_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def loss_of(ids: list[int], labels: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        if not cfg.get("chunked_lm_loss"):
            return model(input_ids=x, attention_mask=torch.ones_like(x), labels=y).loss

        # Với chuỗi dài, logits [seq, vocab] chiếm vài GiB dù transformer đã
        # gradient-checkpoint. Tính cross-entropy theo lát và checkpoint lm_head
        # để chỉ tái tạo một lát logits trong backward; không cắt mất câu trả lời.
        causal_lm = model.get_base_model()
        hidden = causal_lm.model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        shifted = y[:, 1:]
        active = (shifted != IGNORE).sum().clamp_min(1)
        chunk_tokens = int(cfg.get("loss_chunk_tokens", 256))

        def chunk_loss(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = causal_lm.lm_head(h).float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=IGNORE,
                reduction="sum",
            )

        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[1] - 1, chunk_tokens):
            end = min(hidden.shape[1] - 1, start + chunk_tokens)
            h = hidden[:, start:end, :]
            target = y[:, start + 1 : end + 1]
            piece = (
                checkpoint(chunk_loss, h, target, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(h, target)
            )
            total = total + piece
        return total / active

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        val_limit = 2 if args.smoke else int(cfg.get("validation_max_examples", len(data["val"])))
        subset = data["val"][:val_limit]
        losses = [loss_of(*ex).item() for ex in subset]
        model.train()
        return sum(losses) / len(losses)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    adapter_id = f"{'smoke-' if args.smoke else ''}{method}-{cfg['name']}-{stamp}"
    out_dir = Path(cfg["output_root"]) / adapter_id
    history: list[dict[str, Any]] = []
    best = float("inf")
    patience = 0
    base_val = val_loss()
    history.append({"step": 0, "val_loss": round(base_val, 4)})
    print(f"step 0 val_loss={base_val:.4f}")
    model.train()
    order = list(range(len(data["train"])))
    step, micro, t0 = 0, 0, time.time()
    running: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    while step < max_steps:
        random.shuffle(order)
        for idx in order:
            loss = loss_of(*data["train"][idx]) / cfg["grad_accum"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn tại step {step}")
            loss.backward()
            running.append(loss.item() * cfg["grad_accum"])
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec: dict[str, Any] = {"step": step, "train_loss": round(sum(running) / len(running), 4),
                                   "lr": sched.get_last_lr()[0]}
            running = []
            if step % cfg["eval_every_steps"] == 0 or step == max_steps:
                vl = val_loss()
                rec["val_loss"] = round(vl, 4)
                if vl < best - 1e-4:
                    best, patience = vl, 0
                    model.save_pretrained(out_dir)
                    rec["saved"] = True
                else:
                    patience += 1
            history.append(rec)
            print(json.dumps(rec), flush=True)
            if step >= max_steps or patience >= cfg.get("early_stopping_patience", 99):
                break
        if patience >= cfg.get("early_stopping_patience", 99):
            print("early_stopping")
            break

    tok.save_pretrained(out_dir)
    meta = {
        "adapter_id": adapter_id,
        "method": method,
        "base_model": cfg["base_model"],
        "base_revision": cfg["base_revision"],
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest": manifest,
        "config": cfg,
        "trainable_params": trainable,
        "total_params": total,
        "steps": step,
        "best_val_loss": best,
        "base_val_loss": base_val,
        "history": history,
        "validation_examples_per_check": 2 if args.smoke else min(
            len(data["val"]), int(cfg.get("validation_max_examples", len(data["val"])))
        ),
        "masking_example": {k: mask_check[k] for k in ("tokens", "trained_tokens")},
        "dropped_too_long": dropped,
        "train_seconds": round(time.time() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "library_versions": {p: md.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
        "finished_at": datetime.now(UTC).isoformat(),
        "synthetic_data": True,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: meta[k] for k in ("adapter_id", "steps", "base_val_loss", "best_val_loss",
                                            "train_seconds", "peak_vram_gb")}, indent=1))

    # nạp lại adapter vừa lưu lên base sạch trong cùng tiến trình (kiểm tra lưu/nạp); tiến trình mới: compare
    model = opt = None  # giải phóng VRAM trước khi nạp lại
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["base_revision"],
                                                dtype=torch.bfloat16, device_map="cuda")
    reloaded = PeftModel.from_pretrained(base, str(out_dir))
    reloaded.eval()
    ids, labels = data["val"][0]
    with torch.no_grad():
        x = torch.tensor([ids], device="cuda")
        rl = reloaded(input_ids=x, labels=torch.tensor([labels], device="cuda")).loss.item()
    print(f"reload_check val[0] loss={rl:.4f} (finite={math.isfinite(rl)})")
    return 0 if math.isfinite(rl) else 1


if __name__ == "__main__":
    raise SystemExit(main())
