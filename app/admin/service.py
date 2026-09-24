"""Nghiệp vụ quản trị (dùng chung cho Admin API và CLI ``botctl``).

Vòng cải thiện dữ liệu:
  lượt hội thoại -> quản trị viên chấm (good/bad) hoặc viết câu trả lời sửa -> tạo MẪU CHỜ DUYỆT
  (đã che PII) -> quản trị viên duyệt + xác nhận quyền sử dụng -> xuất phiên bản dữ liệu có lineage.
Không mẫu nào tự động vào dữ liệu huấn luyện; câu trả lời bot không tự coi là nhãn đúng.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.config.settings import PROJECT_ROOT
from app.observability.redaction import contains_pii, redact_pii
from app.observability.tracing import Tracer
from app.storage.db import session_scope
from app.storage.repository import audit

CONTEXT_MESSAGES = 8
ROLE_TO_TRAIN = {"user": "user", "assistant": "assistant", "human_agent": "assistant"}


class NotFound(Exception):  # noqa: N818
    pass


class InvalidAction(Exception):  # noqa: N818
    pass


# ----------------------------------------------------------------------------- xem và chấm
async def list_turns(*, status: str = "unrated", limit: int = 20) -> list[dict[str, Any]]:
    where = {
        "unrated": "NOT EXISTS (SELECT 1 FROM feedback f WHERE f.turn_id = t.id)",
        "flagged": "jsonb_array_length(COALESCE(t.check_flags, '[]'::jsonb)) > 0",
        "bad": "EXISTS (SELECT 1 FROM feedback f WHERE f.turn_id = t.id AND f.rating = 'bad')",
        "errors": "(t.outcome LIKE 'fallback%' OR t.outcome LIKE 'failed%' OR t.error IS NOT NULL)",
        "all": "true",
    }.get(status)
    if where is None:
        raise InvalidAction(f"status không hợp lệ: {status}")
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(
                        f"""
                    SELECT t.id, t.conversation_id, t.created_at, t.mode, t.outcome, t.model_name, t.adapter_version,
                           t.prompt_version, t.check_flags, t.total_latency_ms,
                           (SELECT string_agg(COALESCE(m.text, '[đính kèm]'), E'\\n' ORDER BY m.id) FROM messages m
                              WHERE m.turn_id = t.id AND m.role = 'user') AS user_text,
                           (SELECT string_agg(m.text, E'\\n' ORDER BY m.chunk_index) FROM messages m
                              WHERE m.turn_id = t.id AND m.role = 'assistant') AS bot_text,
                           (SELECT f.rating FROM feedback f WHERE f.turn_id = t.id ORDER BY f.id DESC LIMIT 1) AS rating
                    FROM turns t
                    WHERE t.status <> 'running' AND {where}
                    ORDER BY t.created_at DESC LIMIT :lim
                    """  # noqa: S608 - where lấy từ dict cố định
                    ),
                    {"lim": limit},
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


async def rate_turn(
    turn_id: uuid.UUID, rating: str, reviewer: str, notes: str | None, tracer: Tracer | None = None
) -> dict[str, Any]:
    if rating not in ("good", "bad"):
        raise InvalidAction("rating phải là good hoặc bad")
    async with session_scope() as s:
        turn = (await s.execute(text("SELECT id, trace_id FROM turns WHERE id = :t"), {"t": turn_id})).first()
        if turn is None:
            raise NotFound("turn không tồn tại")
        fid = (
            await s.execute(
                text(
                    "INSERT INTO feedback (turn_id, rating, notes, reviewer) VALUES (:t, :r, :n, :rv) RETURNING id"
                ),
                {"t": turn_id, "r": rating, "n": notes, "rv": reviewer},
            )
        ).scalar_one()
        cand = None
        if rating == "good":
            cand = await _create_candidate(s, turn_id, source="admin_good_rating", target=None, rejected=None)
        await audit(s, reviewer, "rate_turn", str(turn_id), {"rating": rating, "candidate_id": cand})
    if tracer is not None:
        tracer.score(
            trace_id=turn[1], name="admin_rating", value=1.0 if rating == "good" else 0.0, data_type="BOOLEAN"
        )
    return {"feedback_id": fid, "candidate_id": cand}


async def correct_turn(
    turn_id: uuid.UUID, corrected_reply: str, reviewer: str, notes: str | None, tracer: Tracer | None = None
) -> dict[str, Any]:
    corrected_reply = (corrected_reply or "").strip()
    if not corrected_reply:
        raise InvalidAction("câu trả lời sửa đang rỗng")
    async with session_scope() as s:
        turn = (await s.execute(text("SELECT id, trace_id FROM turns WHERE id = :t"), {"t": turn_id})).first()
        if turn is None:
            raise NotFound("turn không tồn tại")
        original = (
            await s.execute(
                text(
                    """SELECT string_agg(text, E'\\n' ORDER BY chunk_index) FROM messages
                       WHERE turn_id = :t AND role = 'assistant'"""
                ),
                {"t": turn_id},
            )
        ).scalar_one_or_none()
        fid = (
            await s.execute(
                text(
                    """INSERT INTO feedback (turn_id, rating, corrected_reply, notes, reviewer)
                       VALUES (:t, 'bad', :c, :n, :rv) RETURNING id"""
                ),
                {"t": turn_id, "c": corrected_reply, "n": notes, "rv": reviewer},
            )
        ).scalar_one()
        cand = await _create_candidate(
            s, turn_id, source="admin_correction", target=corrected_reply, rejected=original
        )
        await audit(s, reviewer, "correct_turn", str(turn_id), {"candidate_id": cand})
    if tracer is not None:
        tracer.score(
            trace_id=turn[1], name="admin_rating", value=0.0, data_type="BOOLEAN", comment="corrected"
        )
    return {"feedback_id": fid, "candidate_id": cand}


async def _create_candidate(
    s, turn_id: uuid.UUID, *, source: str, target: str | None, rejected: str | None
) -> int | None:
    conv_id = (
        await s.execute(text("SELECT conversation_id FROM turns WHERE id = :t"), {"t": turn_id})
    ).scalar_one()
    turn_msgs = (
        (
            await s.execute(
                text(
                    """SELECT id, role, text, chunk_index FROM messages WHERE turn_id = :t
                   AND ((role = 'user') OR (role = 'assistant' AND status IN ('sent', 'uncertain', 'dry_run')))
                   ORDER BY role DESC, id"""
                ),
                {"t": turn_id},
            )
        )
        .mappings()
        .all()
    )
    user_msgs = [m for m in turn_msgs if m["role"] == "user" and m["text"]]
    bot_text = "\n".join(m["text"] for m in turn_msgs if m["role"] == "assistant" and m["text"])
    if not user_msgs:
        return None
    first_seq = min(m["id"] for m in user_msgs)
    ctx_rows = (
        (
            await s.execute(
                text(
                    """SELECT role, text FROM messages WHERE conversation_id = :cid AND id < :first AND text IS NOT NULL
                   AND ((role = 'user' AND status IN ('consumed','skipped'))
                        OR (role = 'assistant' AND status IN ('sent','uncertain')) OR role = 'human_agent')
                   ORDER BY id DESC LIMIT :n"""
                ),
                {"cid": conv_id, "first": first_seq, "n": CONTEXT_MESSAGES},
            )
        )
        .mappings()
        .all()
    )
    messages = [{"role": ROLE_TO_TRAIN[r["role"]], "content": r["text"]} for r in reversed(ctx_rows)]
    messages.append({"role": "user", "content": "\n".join(m["text"] for m in user_msgs)})
    final_target = target if target is not None else bot_text
    if not final_target:
        return None
    messages.append({"role": "assistant", "content": final_target})
    merged: list[dict[str, str]] = []  # gộp các lượt liên tiếp cùng vai (chat template cần xen kẽ)
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    had_pii = any(contains_pii(m["content"]) for m in merged) or bool(rejected and contains_pii(rejected))
    sample = {
        "messages": [{"role": m["role"], "content": redact_pii(m["content"])} for m in merged],
        "language": "vi",
    }
    return (
        await s.execute(
            text(
                """INSERT INTO training_candidates (turn_id, conversation_id, source, status, sample, rejected_reply,
                       pii_redacted, pii_detected_before_redaction)
                   VALUES (:t, :cid, :src, 'pending_review', CAST(:sample AS JSONB), :rej, true, :pii)
                   RETURNING id"""
            ),
            {
                "t": turn_id,
                "cid": conv_id,
                "src": source,
                "sample": json.dumps(sample, ensure_ascii=False),
                "rej": redact_pii(rejected) if rejected else None,
                "pii": had_pii,
            },
        )
    ).scalar_one()


# ----------------------------------------------------------------------------- duyệt mẫu
async def list_candidates(status: str = "pending_review", limit: int = 50) -> list[dict[str, Any]]:
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(
                        """SELECT id, turn_id, source, status, sample, rejected_reply, pii_detected_before_redaction,
                              usage_rights_confirmed, reviewer, reviewed_at, exported_versions, created_at
                       FROM training_candidates WHERE status = :st ORDER BY id DESC LIMIT :lim"""
                    ),
                    {"st": status, "lim": limit},
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


async def review_candidate(
    cand_id: int,
    *,
    approve: bool,
    reviewer: str,
    usage_rights_confirmed: bool,
    notes: str | None,
    edited_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if approve and not usage_rights_confirmed:
        raise InvalidAction("Phải xác nhận quyền sử dụng dữ liệu (usage_rights_confirmed=true) để duyệt")
    async with session_scope() as s:
        row = (
            await s.execute(text("SELECT id, status FROM training_candidates WHERE id = :i"), {"i": cand_id})
        ).first()
        if row is None:
            raise NotFound("mẫu không tồn tại")
        params: dict[str, Any] = {
            "i": cand_id,
            "st": "approved" if approve else "rejected",
            "rv": reviewer,
            "u": usage_rights_confirmed,
            "n": notes,
        }
        extra = ""
        if edited_sample is not None:
            msgs = edited_sample.get("messages")
            if not isinstance(msgs, list) or not msgs or msgs[-1].get("role") != "assistant":
                raise InvalidAction("sample.messages phải kết thúc bằng lượt assistant")
            edited_sample = {
                **edited_sample,
                "messages": [{"role": m["role"], "content": redact_pii(str(m["content"]))} for m in msgs],
            }
            extra = ", sample = CAST(:sample AS JSONB)"
            params["sample"] = json.dumps(edited_sample, ensure_ascii=False)
        await s.execute(
            text(
                f"""UPDATE training_candidates SET status = :st, reviewer = :rv, reviewed_at = now(),
                       usage_rights_confirmed = :u, review_notes = :n{extra} WHERE id = :i"""  # noqa: S608
            ),
            params,
        )
        await audit(s, reviewer, "approve_candidate" if approve else "reject_candidate", str(cand_id), None)
    return {"id": cand_id, "status": params["st"]}


# ----------------------------------------------------------------------------- xuất dữ liệu (lineage)
async def export_approved(
    *, reviewer: str, out_root: Path | None = None, kind: str = "sft"
) -> dict[str, Any]:
    """Xuất mẫu approved + usage_rights_confirmed + pii_redacted thành một phiên bản dữ liệu bất biến."""
    if kind not in ("sft", "dpo"):
        raise InvalidAction("kind phải là sft hoặc dpo")
    out_root = out_root or PROJECT_ROOT / "data" / "exports"
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(
                        """SELECT id, source, sample, rejected_reply FROM training_candidates
                       WHERE status = 'approved' AND usage_rights_confirmed AND pii_redacted ORDER BY id"""
                    )
                )
            )
            .mappings()
            .all()
        )
        records: list[dict[str, Any]] = []
        for r in rows:
            msgs = r["sample"]["messages"]
            if kind == "sft":
                records.append(
                    {
                        "conversation_id": f"fanpage-candidate-{r['id']}",
                        "group_id": f"fanpage-candidate-{r['id']}",
                        "messages": msgs,
                        "source": f"fanpage_feedback:{r['source']}",
                        "license": "fanpage-owner-approved",
                        "language": r["sample"].get("language", "vi"),
                        "approved_for_training": True,
                        "synthetic": False,
                    }
                )
            elif r["source"] == "admin_correction" and r["rejected_reply"]:
                # DPO: chosen = câu sửa của quản trị viên, rejected = câu bot bị sửa. KHÔNG tạo từ thumbs-up đơn lẻ.
                records.append(
                    {
                        "pair_id": f"fanpage-pair-{r['id']}",
                        "prompt": msgs[:-1],
                        "chosen": msgs[-1]["content"],
                        "rejected": r["rejected_reply"],
                        "source": "fanpage_feedback:admin_correction",
                        "language": r["sample"].get("language", "vi"),
                    }
                )
        if not records:
            raise InvalidAction("Chưa có mẫu nào đủ điều kiện xuất")
        body = "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in records) + "\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        version = f"fanpage-{kind}-{datetime.now(UTC):%Y%m%d%H%M}-{digest[:8]}"
        out = out_root / version
        out.mkdir(parents=True, exist_ok=False)
        (out / "data.jsonl").write_text(body, encoding="utf-8")
        ids = [r["id"] for r in rows]
        manifest = {
            "dataset_version": version,
            "kind": kind,
            "created_at": datetime.now(UTC).isoformat(),
            "num_records": len(records),
            "candidate_ids": ids,
            "sha256": digest,
            "exported_by": reviewer,
            "policy": "only approved + usage_rights_confirmed + pii_redacted candidates",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        await s.execute(
            text(
                """INSERT INTO dataset_exports (version, path, sha256, num_samples, candidate_ids, kind, notes)
                   VALUES (:v, :p, :h, :n, CAST(:ids AS JSONB), :k, :notes)"""
            ),
            {
                "v": version,
                "p": str(out),
                "h": digest,
                "n": len(records),
                "ids": json.dumps(ids),
                "k": kind,
                "notes": f"exported by {reviewer}",
            },
        )
        await s.execute(
            text(
                """UPDATE training_candidates SET exported_versions =
                       COALESCE(exported_versions, '[]'::jsonb) || CAST(:v AS JSONB)
                   WHERE id = ANY(:ids)"""
            ),
            {"v": json.dumps([version]), "ids": ids},
        )
        await audit(s, reviewer, "export_dataset", version, {"records": len(records), "kind": kind})
    return {"dataset_version": version, "path": str(out), "records": len(records)}


# ----------------------------------------------------------------------------- handoff
async def get_conversation(conv_id: uuid.UUID) -> dict[str, Any]:
    async with session_scope() as s:
        r = (
            (
                await s.execute(
                    text(
                        """SELECT id, page_id, user_ref, created_at, last_user_message_at, last_bot_message_at,
                              handoff_active, handoff_reason, handoff_by, handoff_changed_at, bot_resumed_at,
                              summary IS NOT NULL AS has_summary, next_run_at, attempts, last_error
                       FROM conversations WHERE id = :cid"""
                    ),
                    {"cid": conv_id},
                )
            )
            .mappings()
            .first()
        )
    if r is None:
        raise NotFound("conversation không tồn tại")
    return dict(r)


async def list_conversations(
    limit: int = 20, handoff_only: bool = False, include_psid: bool = False
) -> list[dict[str, Any]]:
    cols = "id, user_ref, last_user_message_at, handoff_active, handoff_by" + (
        ", psid" if include_psid else ""
    )
    cond = "WHERE handoff_active" if handoff_only else ""
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(f"SELECT {cols} FROM conversations {cond} ORDER BY updated_at DESC LIMIT :lim"),  # noqa: S608
                    {"lim": limit},
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


async def set_handoff(
    conv_id: uuid.UUID, active: bool, actor: str, reason: str | None = None
) -> dict[str, Any]:
    async with session_scope() as s:
        if active:
            res = await s.execute(
                text(
                    """UPDATE conversations SET handoff_active = true, handoff_reason = :r, handoff_by = :a,
                           handoff_changed_at = now() WHERE id = :cid RETURNING id"""
                ),
                {"cid": conv_id, "r": reason or "admin", "a": actor},
            )
        else:
            # bật lại bot: lượt trả lời kế tiếp sẽ nhắc lại rằng đây là trợ lý AI (chính sách Messenger)
            res = await s.execute(
                text(
                    """UPDATE conversations SET handoff_active = false, handoff_reason = NULL, handoff_by = :a,
                           handoff_changed_at = now(), bot_resumed_at = now() WHERE id = :cid RETURNING id"""
                ),
                {"cid": conv_id, "a": actor},
            )
        if res.scalar_one_or_none() is None:
            raise NotFound("conversation không tồn tại")
        await audit(s, actor, "handoff_on" if active else "handoff_off", str(conv_id), {"reason": reason})
    return await get_conversation(conv_id)


# ----------------------------------------------------------------------------- xóa dữ liệu
async def delete_conversation(
    conv_id: uuid.UUID, actor: str, checkpointer: Any, tracer: Tracer | None = None
) -> dict[str, Any]:
    """Xóa tin nhắn, tóm tắt, turn, phản hồi, mẫu huấn luyện, checkpoint LangGraph và trace Langfuse.

    Lưu ý: KHÔNG thể xóa ảnh hưởng của dữ liệu đã dùng để huấn luyện adapter; audit log ghi lại các
    phiên bản dữ liệu bị ảnh hưởng để quyết định huấn luyện lại.
    """
    async with session_scope() as s:
        exists = (
            await s.execute(text("SELECT user_ref FROM conversations WHERE id = :cid"), {"cid": conv_id})
        ).first()
        if exists is None:
            raise NotFound("conversation không tồn tại")
        turns = (
            await s.execute(
                text("SELECT id, graph_thread_id, trace_id FROM turns WHERE conversation_id = :cid"),
                {"cid": conv_id},
            )
        ).all()
        affected = (
            (
                await s.execute(
                    text(
                        """SELECT DISTINCT jsonb_array_elements_text(exported_versions) FROM training_candidates
                       WHERE conversation_id = :cid AND exported_versions IS NOT NULL"""
                    ),
                    {"cid": conv_id},
                )
            )
            .scalars()
            .all()
        )
    threads = [t[1] for t in turns if t[1]]
    for th in threads:  # xóa checkpoint trước (idempotent)
        await checkpointer.adelete_thread(th)
    traces_deleted = (
        tracer.delete_traces([t[2] for t in turns if t[2]]) if tracer and tracer.enabled else False
    )
    async with session_scope() as s:
        n_cand = len(
            (
                await s.execute(
                    text("DELETE FROM training_candidates WHERE conversation_id = :cid RETURNING id"),
                    {"cid": conv_id},
                )
            ).all()
        )
        n_msg = len(
            (
                await s.execute(
                    text("DELETE FROM messages WHERE conversation_id = :cid RETURNING id"), {"cid": conv_id}
                )
            ).all()
        )
        n_turn = len(
            (
                await s.execute(
                    text("DELETE FROM turns WHERE conversation_id = :cid RETURNING id"), {"cid": conv_id}
                )
            ).all()
        )
        await s.execute(text("DELETE FROM conversations WHERE id = :cid"), {"cid": conv_id})
        details = {
            "user_ref": exists[0],
            "messages": n_msg,
            "turns": n_turn,
            "candidates": n_cand,
            "checkpoint_threads": len(threads),
            "langfuse_traces_deleted": traces_deleted,
            "affected_dataset_versions": list(affected),
        }
        await audit(s, actor, "delete_conversation", str(conv_id), details)
    return details
