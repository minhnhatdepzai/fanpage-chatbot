// Xử lý văn bản thuần (không đụng DOM) -> test được bằng `node --test`.

export type Segment = { kind: "text"; value: string } | { kind: "link"; value: string; href: string };

const URL_RE = /https?:\/\/[^\s<>"'`)\]]+/g;
const TRAILING = /[.,;:!?]+$/;

/** Tách văn bản thành đoạn chữ và đường link http(s). Không bao giờ trả về HTML. */
export function linkify(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(URL_RE)) {
    let url = m[0];
    const trail = url.match(TRAILING)?.[0] ?? "";
    url = url.slice(0, url.length - trail.length);
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", value: text.slice(last, start) });
    if (isSafeHttpUrl(url)) out.push({ kind: "link", value: url, href: url });
    else out.push({ kind: "text", value: url });
    last = start + url.length;
  }
  if (last < text.length) out.push({ kind: "text", value: text.slice(last) });
  return out;
}

export function isSafeHttpUrl(value: string): boolean {
  try {
    const u = new URL(value);
    return u.protocol === "https:" || u.protocol === "http:";
  } catch {
    return false;
  }
}

/** Chuẩn hóa tin người dùng trước khi gửi: bỏ khoảng trắng thừa, giới hạn độ dài. */
export function prepareOutgoing(text: string, maxChars: number): string | null {
  const t = text.replace(/\r\n?/g, "\n").trim();
  if (!t || t.length > maxChars) return null;
  return t;
}
