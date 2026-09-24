"""So sánh baseline (model gốc) với adapter trên bộ kịch bản đánh giá, qua ĐÚNG pipeline production.

    uv run python -m evaluation.scripts.compare --adapter artifacts/adapters/<id> [--adapter ...]

Chạy in-process (Transformers + PEFT, hỗ trợ cả VeRA) -> tắt model server trước (cùng GPU). Mỗi kịch bản dùng seed
cố định nên baseline và adapter được so trong cùng điều kiện. Ghi:
  - evaluation/results/<run>/report.json (+ report của từng adapter, có "gate" cho botctl model promote)
  - evaluation/results/<run>/outputs.jsonl (câu trả lời để người duyệt đọc lại; thư mục này không commit)
Chấm bằng luật (không dùng LLM-as-judge); vẫn cần người đọc lại mẫu trước khi promote.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import load_knowledge
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.output_check import _UNCERTAIN
from app.conversation.profile import load_profile
from app.conversation.prompts import SOURCES_FOOTER_TITLE, UNVERIFIED_NOTE
from app.observability.redaction import strip_diacritics
from app.providers.llm import LLMResult

HALLUCINATION_FLAGS = {
    "unverified_url_removed",
    "invalid_citation_removed",
    "false_verification_claim_removed",
    "model_source_line_removed",
}


class RuntimeProvider:
    """ChatProvider chạy thẳng ModelRuntime (không qua HTTP) với seed đặt theo kịch bản."""

    name = "runtime"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.seed = 0
        self.latencies: list[int] = []

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        role = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        msgs = [{"role": role[type(m)], "content": str(m.content)} for m in messages]
        res = await asyncio.to_thread(
            self.runtime.generate, msgs, max_new_tokens=max_tokens, temperature=temperature, seed=self.seed
        )
        self.latencies.append(res.latency_ms)
        return LLMResult(res.text, "runtime", self.runtime.base_model_id, self.runtime.adapter_id,
                         res.prompt_tokens, res.completion_tokens, res.latency_ms)


class HttpProvider:
    """ChatProvider gọi endpoint OpenAI-compatible (model server, Ollama...) với seed cố định theo kịch bản."""

    def __init__(self, label: str, url: str, model: str, api_key: str | None) -> None:
        self.name = label
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.seed = 0
        self.latencies: list[int] = []

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        import httpx

        role = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        body = {"model": self.model, "messages": [{"role": role[type(m)], "content": str(m.content)} for m in messages],
                "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.8, "seed": self.seed}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=max(timeout, 120)) as c:
            r = await c.post(self.url, json=body, headers=headers)
            r.raise_for_status()
            d = r.json()
        ms = int((time.perf_counter() - t0) * 1000)
        self.latencies.append(ms)
        u = d.get("usage") or {}
        return LLMResult(d["choices"][0]["message"]["content"] or "", self.name, d.get("model", self.model), None,
                         u.get("prompt_tokens"), u.get("completion_tokens"), ms)


def _body(text: str) -> str:
    """Phần do model viết: bỏ danh sách nguồn và ghi chú do hệ thống gắn."""
    text = text.replace(UNVERIFIED_NOTE, "")
    return text.split(f"\n\n{SOURCES_FOOTER_TITLE}\n", 1)[0].strip()


def score(sc: dict[str, Any], reply: str, flags: list[str]) -> dict[str, Any]:
    body = _body(reply)
    checks: dict[str, bool] = {}
    if sc.get("cite"):
        checks["cite"] = "sources_attached" in flags
    if sc.get("uncertain"):
        checks["uncertain"] = bool(_UNCERTAIN.search(strip_diacritics(body.lower())))
    if sc.get("no_note"):
        checks["no_note"] = "unverified_note_added" not in flags
    if sc.get("include_any"):
        checks["include_any"] = any(s.lower() in reply.lower() for s in sc["include_any"])
    if sc.get("exclude"):
        checks["exclude"] = not any(s.lower() in reply.lower() for s in sc["exclude"])
    halluc = sorted(HALLUCINATION_FLAGS & set(flags))
    checks["no_hallucination_signal"] = not halluc
    return {"passed": all(checks.values()), "checks": checks, "hallucination_flags": halluc,
            "chars": len(body), "sentences": len([x for x in re.split(r"(?<=[.!?…])\s+", body) if x.strip()])}


async def run_suite(provider: RuntimeProvider, scenarios: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    s = get_settings()
    kb = load_knowledge(s.knowledge_dir)
    profile = load_profile(s.fanpage_profile_path)
    graph = build_graph()
    results = []
    for sc in scenarios:
        store = InMemoryConversationStore(now=lambda: datetime.now(UTC))
        ctx = GraphContext(settings=s, store=store, llm=provider, profile=profile, knowledge=kb)
        cid = store.ensure_conversation("EVAL", f"{label}-{sc['id']}")
        result: dict[str, Any] = {}
        for i, text in enumerate(sc["turns"]):
            provider.seed = int(hashlib.sha256(f"{sc['id']}:{i}".encode()).hexdigest()[:8], 16)
            store.add_user_message(cid, text, ts=datetime.now(UTC))
            tid = store.begin_turn(cid)
            result = await run_turn(graph, turn_id=tid, conversation_id=cid, context=ctx)
            store.apply_result(cid, tid, result)
        reply = "\n".join(result.get("reply_parts") or [])
        flags = list(result.get("check_flags") or [])
        results.append({"id": sc["id"], "category": sc["category"], "turns": sc["turns"], "reply": reply,
                        "flags": flags, "hits": [h["id"] for h in result.get("knowledge_hits") or []],
                        **score(sc, reply, flags)})
    return results


def summarize(results: list[dict[str, Any]], latencies: list[int]) -> dict[str, Any]:
    cats: dict[str, list[bool]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["passed"])
    sourced = [r for r in results if r["category"] == "sourced"]
    lat = sorted(latencies) or [0]
    return {
        "n": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 4),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(cats.items())},
        "hallucination_signals": sum(len(r["hallucination_flags"]) for r in results),
        "sourced_citation_rate": round(sum(r["checks"].get("cite", False) for r in sourced) / max(1, len(sourced)), 4),
        "avg_chars": round(statistics.mean(r["chars"] for r in results), 1),
        "latency_ms_p50": lat[len(lat) // 2],
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "latency_samples": len(latencies),
    }


def gate(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Chỉ đạt khi TỐT HƠN rõ (>= 2 kịch bản) và không tệ hơn ở chỉ số an toàn nào."""
    reasons = []
    if cand["passed"] < base["passed"] + 2:
        reasons.append(f"số kịch bản đạt {cand['passed']} < baseline {base['passed']} + 2")
    if cand["hallucination_signals"] > base["hallucination_signals"]:
        reasons.append("nhiều tín hiệu bịa hơn baseline")
    if cand["sourced_citation_rate"] < base["sourced_citation_rate"]:
        reasons.append("tỉ lệ trích nguồn thấp hơn baseline")
    return {"passed": not reasons, "reasons": reasons, "baseline": base, "candidate": cand}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], type=Path)
    ap.add_argument("--scenarios", type=Path, default=Path("evaluation/datasets/edu_ent_eval_v1.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--http", action="append", default=[], metavar="NHÃN=URL|MODEL",
                    help="đánh giá qua HTTP (OpenAI-compatible); mục đầu tiên là baseline, vd. "
                         "bf16=http://127.0.0.1:8100/v1|qwen3-4b-instruct-2507@base")
    args = ap.parse_args()
    if args.http:
        return http_mode(args)

    import torch

    from serving.runtime import ModelRuntime

    s = get_settings()
    scenarios = yaml.safe_load(args.scenarios.read_text())["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    run_dir = Path("evaluation/results") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = ModelRuntime(s.base_model_id, s.base_model_revision)
    provider = RuntimeProvider(runtime)
    all_rows: list[dict[str, Any]] = []
    t0 = time.time()
    base_rows = asyncio.run(run_suite(provider, scenarios, "base"))
    base = summarize(base_rows, provider.latencies)
    base["vram_gb"] = runtime.vram_gb()
    all_rows += [{"model": "base", **r} for r in base_rows]
    report: dict[str, Any] = {"scenarios": str(args.scenarios), "n": len(scenarios), "baseline": base,
                              "settings": {"temperature": s.llm_temperature, "max_tokens": s.llm_max_output_tokens,
                                           "grounding_mode": s.grounding_mode.value},
                              "candidates": {}}
    print("baseline", json.dumps(base, ensure_ascii=False))
    for path in args.adapter:
        runtime.load_adapter(str(path), path.name)
        provider.latencies = []
        rows = asyncio.run(run_suite(provider, scenarios, path.name))
        summ = summarize(rows, provider.latencies)
        summ["vram_gb"] = runtime.vram_gb()
        g = gate(base, summ)
        report["candidates"][path.name] = g
        (run_dir / f"report-{path.name}.json").write_text(
            json.dumps({"adapter": path.name, "gate": g, "scenarios": str(args.scenarios)}, ensure_ascii=False, indent=2)
        )
        all_rows += [{"model": path.name, **r} for r in rows]
        print(path.name, json.dumps(summ, ensure_ascii=False), "gate:", g["passed"], g["reasons"])
        runtime.unload_adapter()
        torch.cuda.empty_cache()
    report["seconds"] = round(time.time() - t0, 1)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (run_dir / "outputs.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"run_dir={run_dir}")
    return 0


def http_mode(args: argparse.Namespace) -> int:
    s = get_settings()
    scenarios = yaml.safe_load(args.scenarios.read_text())["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    run_dir = Path("evaluation/results") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    own = s.llm_base_url.rstrip("/")
    report: dict[str, Any] = {"scenarios": str(args.scenarios), "n": len(scenarios), "mode": "http", "targets": {}}
    rows_all: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for spec in args.http:
        label, rest = spec.split("=", 1)
        url, model = rest.split("|", 1)
        key = s.model_server_api_key.get_secret_value() if url.rstrip("/") == own else None  # chỉ gửi khóa cho server của mình
        prov = HttpProvider(label, url, model, key)
        rows = asyncio.run(run_suite(prov, scenarios, label))  # type: ignore[arg-type]
        summ = summarize(rows, prov.latencies)
        if base is None:
            base = summ
            report["baseline"] = {"label": label, **summ}
        else:
            report["targets"][label] = gate(base, summ)
        rows_all += [{"model": label, **r} for r in rows]
        print(label, json.dumps(summ, ensure_ascii=False))
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (run_dir / "outputs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
