"""Pipeline nạp tài liệu: đọc -> (OCR trang scan) -> làm sạch -> kiểm tra dữ liệu cá nhân -> chia đoạn -> embedding -> pgvector.

uv run botctl docs ingest quy-che.pdf --visibility internal
uv run botctl docs ingest bang-gia.docx --visibility public --source "https://example.vn/bang-gia"
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.observability.redaction import contains_pii
from app.rag import store
from app.rag.embedder import Embedder
from app.rag.parsing import DocumentError, chunk_pages, clean_text, parse_file

OcrFn = Callable[[Any], Awaitable[str]]  # ảnh PIL -> văn bản


async def ingest_file(
    path: Path,
    *,
    settings: Settings,
    embedder: Embedder,
    visibility: str,
    actor: str,
    title: str | None = None,
    source: str | None = None,
    ocr: OcrFn | None = None,
    allow_pii_public: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    if visibility not in ("public", "internal"):
        raise DocumentError("visibility phải là public hoặc internal")
    doc = parse_file(path, max_bytes=settings.docs_max_file_mb * 1_000_000, max_pages=settings.docs_max_pages)
    existing = await store.find_document(doc.sha256, visibility)
    if existing and not replace:
        return {"status": "skipped_duplicate", "document_id": str(existing["id"]), "title": existing["title"]}

    ocr_pages, ocr_skipped = 0, []
    for page in doc.pages:
        if page.image is None:
            continue
        if ocr is None:
            ocr_skipped.append(page.number)
            continue
        page.text = clean_text(await ocr(page.image))
        page.image = None
        ocr_pages += 1

    full_text = "\n".join(p.text for p in doc.pages)
    if not full_text.strip():
        raise DocumentError("Không trích được chữ nào (tài liệu scan cần OCR: bật model server có OCR)")
    if visibility == "public" and contains_pii(full_text) and not allow_pii_public:
        raise DocumentError(
            "Tài liệu có dấu hiệu chứa dữ liệu cá nhân (số điện thoại, email, số thẻ...) nên không được đưa lên kênh "
            "công khai. Kiểm tra lại, hoặc nạp với --visibility internal, hoặc dùng --allow-pii-public nếu chắc chắn."
        )

    chunks = chunk_pages(
        doc.pages, max_chars=settings.chunk_max_chars, overlap_chars=settings.chunk_overlap_chars
    )
    vectors = await embedder.embed([c.text for c in chunks])
    rows = [
        {"index": c.index, "text": c.text, "page_from": c.page_from, "page_to": c.page_to, "embedding": v}
        for c, v in zip(chunks, vectors, strict=True)
    ]
    doc_id = await store.insert_document(
        title=title or doc.title_hint,
        source=source or f"Tài liệu: {path.name}",
        original_name=path.name,
        mime=doc.mime,
        sha256=doc.sha256,
        visibility=visibility,
        pages=len(doc.pages),
        ocr_pages=ocr_pages,
        embedding_model=embedder.model,
        chunks=rows,
        created_by=actor,
        metadata={"ocr_skipped_pages": ocr_skipped, "chars": len(full_text)},
        replace=replace,
    )
    return {
        "status": "ingested",
        "document_id": str(doc_id),
        "title": title or doc.title_hint,
        "visibility": visibility,
        "pages": len(doc.pages),
        "ocr_pages": ocr_pages,
        "ocr_skipped_pages": ocr_skipped,
        "chunks": len(rows),
        "chars": len(full_text),
    }
