// Khung chat nhúng website cho fanpage-chatbot.
//
//   <script src="https://<api>/web/widget.js" data-title="AI Test" data-color="#0866ff" defer></script>
//
// - Không phụ thuộc thư viện ngoài; giao diện trong Shadow DOM để không đụng CSS của trang chủ.
// - Tin nhắn hiển thị bằng textContent (không innerHTML) -> không bị chèn mã; link chỉ nhận http(s).
// - Phiên chat (mã phiên + chữ ký do server cấp) và 50 tin gần nhất lưu trong localStorage của trình duyệt.

import { linkify, prepareOutgoing } from "./text";

type Role = "user" | "bot" | "error";
interface StoredMsg { role: Role; text: string }
interface Session { session_id: string; token: string }
interface ServerMsg { id: number; text: string }
interface WidgetConfig { images: boolean; max_chars: number; max_image_mb: number }

const script = document.currentScript as HTMLScriptElement | null;
const API = (script?.dataset.api || (script?.src ? new URL(script.src).origin : location.origin)).replace(/\/+$/, "");
const TITLE = script?.dataset.title || "Trợ lý AI";
const COLOR = /^#[0-9a-f]{3,8}$/i.test(script?.dataset.color || "") ? script!.dataset.color! : "#0866ff";
const STORE_KEY = `fcb:${API}`;
const POLL_MS = 1500;
const POLL_MAX_MS = 90_000;

const store = {
  load(): { session?: Session; msgs: StoredMsg[]; lastId: number } {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) return { msgs: [], lastId: 0, ...JSON.parse(raw) };
    } catch { /* private mode / bị chặn: chạy không lưu */ }
    return { msgs: [], lastId: 0 };
  },
  save(state: { session?: Session; msgs: StoredMsg[]; lastId: number }): void {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ ...state, msgs: state.msgs.slice(-50) }));
    } catch { /* bỏ qua */ }
  },
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json" }, credentials: "omit" });
  if (!res.ok) {
    let detail = `Lỗi ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch { /* không phải JSON */ }
    throw Object.assign(new Error(detail), { status: res.status });
  }
  return (await res.json()) as T;
}

const CSS = `
:host { all: initial; }
* { box-sizing: border-box; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.fab { position: fixed; right: 20px; bottom: 20px; width: 56px; height: 56px; border-radius: 50%; border: 0;
  background: ${COLOR}; color: #fff; font-size: 26px; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.25); z-index: 2147483000; }
.panel { position: fixed; right: 20px; bottom: 88px; width: 360px; max-width: calc(100vw - 32px); height: 520px;
  max-height: calc(100vh - 120px); background: #fff; color: #1c1e21; border-radius: 14px; display: none; flex-direction: column;
  box-shadow: 0 8px 30px rgba(0,0,0,.25); overflow: hidden; z-index: 2147483000; }
.panel.open { display: flex; }
header { background: ${COLOR}; color: #fff; padding: 12px 14px; display: flex; align-items: center; gap: 8px; }
header .t { font-weight: 600; font-size: 15px; } header .s { font-size: 12px; opacity: .85; }
header button { margin-left: auto; background: transparent; border: 0; color: #fff; font-size: 20px; cursor: pointer; }
.log { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; background: #f5f6f7; }
.m { max-width: 85%; padding: 8px 11px; border-radius: 14px; font-size: 14px; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word; }
.m.user { align-self: flex-end; background: ${COLOR}; color: #fff; }
.m.bot { align-self: flex-start; background: #fff; border: 1px solid #e4e6eb; }
.m.error { align-self: center; background: #fff3cd; color: #664d03; font-size: 13px; }
.m a { color: inherit; text-decoration: underline; }
.typing { align-self: flex-start; font-size: 13px; color: #65676b; padding: 2px 6px; }
form { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #e4e6eb; background: #fff; align-items: flex-end; }
textarea { flex: 1; resize: none; border: 1px solid #ccd0d5; border-radius: 18px; padding: 8px 12px; font-size: 14px; max-height: 96px; }
form button { border: 0; background: ${COLOR}; color: #fff; border-radius: 18px; padding: 8px 14px; cursor: pointer; font-size: 14px; }
form button.icon { background: #e4e6eb; color: #1c1e21; padding: 8px 10px; }
form button:disabled { opacity: .5; cursor: default; }
.note { font-size: 11px; color: #65676b; text-align: center; padding: 4px 8px 8px; background: #fff; }
@media (max-width: 480px) { .panel { right: 8px; left: 8px; width: auto; bottom: 80px; height: calc(100vh - 100px); } }
`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, textContent?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (textContent !== undefined) e.textContent = textContent;
  return e;
}

function mount(): void {
  const host = el("div");
  host.setAttribute("data-fanpage-chatbot", "");
  document.body.appendChild(host);
  const root = host.attachShadow({ mode: "open" });
  const style = el("style");
  style.textContent = CSS;

  const fab = el("button", "fab", "💬");
  fab.setAttribute("aria-label", `Mở khung chat ${TITLE}`);
  const panel = el("section", "panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", TITLE);
  const header = el("header");
  const titles = el("div");
  titles.append(el("div", "t", TITLE), el("div", "s", "Trợ lý AI tự động"));
  const close = el("button", undefined, "×");
  close.setAttribute("aria-label", "Đóng");
  header.append(titles, close);
  const log = el("div", "log");
  log.setAttribute("role", "log");
  log.setAttribute("aria-live", "polite");
  const typing = el("div", "typing", "Đang trả lời…");
  const form = el("form");
  const input = el("textarea");
  input.rows = 1;
  input.placeholder = "Nhập tin nhắn…";
  input.setAttribute("aria-label", "Tin nhắn");
  const attach = el("button", "icon", "🖼");
  attach.type = "button";
  attach.title = "Gửi ảnh";
  attach.setAttribute("aria-label", "Gửi ảnh");
  attach.hidden = true;
  const file = el("input");
  file.type = "file";
  file.accept = "image/png,image/jpeg,image/webp";
  file.hidden = true;
  const send = el("button", undefined, "Gửi");
  send.type = "submit";
  form.append(attach, file, input, send);
  const note = el("div", "note", "Câu trả lời do AI tạo, có thể sai. Nguồn (nếu có) ghi ngay dưới câu trả lời.");
  panel.append(header, log, form, note);
  root.append(style, fab, panel);

  const state = store.load();
  let config: WidgetConfig = { images: false, max_chars: 1000, max_image_mb: 8 };
  let polling = false;

  const render = (m: StoredMsg): void => {
    const b = el("div", `m ${m.role}`);
    for (const seg of linkify(m.text)) {
      if (seg.kind === "link") {
        const a = el("a", undefined, seg.value);
        a.href = seg.href;
        a.target = "_blank";
        a.rel = "noopener noreferrer nofollow";
        b.append(a);
      } else b.append(document.createTextNode(seg.value));
    }
    log.append(b);
    log.scrollTop = log.scrollHeight;
  };
  const push = (m: StoredMsg, persist = true): void => {
    render(m);
    if (persist && m.role !== "error") {
      state.msgs.push(m);
      store.save(state);
    }
  };
  const setTyping = (on: boolean): void => {
    if (on) log.append(typing);
    else typing.remove();
    log.scrollTop = log.scrollHeight;
  };

  async function ensureSession(): Promise<Session> {
    if (!state.session) {
      state.session = await api<Session>("/web/session", { method: "POST", body: "{}" });
      store.save(state);
    }
    return state.session;
  }

  async function poll(): Promise<void> {
    if (polling || !state.session) return;
    polling = true;
    setTyping(true);
    const started = Date.now();
    try {
      while (Date.now() - started < POLL_MAX_MS) {
        const q = new URLSearchParams({ session_id: state.session.session_id, token: state.session.token, after: String(state.lastId) });
        const res = await api<{ messages: ServerMsg[]; pending: boolean }>(`/web/messages?${q}`);
        for (const m of res.messages) {
          state.lastId = Math.max(state.lastId, m.id);
          typing.remove();
          push({ role: "bot", text: m.text });
          if (res.pending) setTyping(true);
        }
        store.save(state);
        // không còn gì đang xử lý: dừng khi đã nhận trả lời, hoặc sau vài giây nếu chỉ mở lại khung chat
        if (!res.pending && (res.messages.length || Date.now() - started > 4000)) break;
        await new Promise((r) => setTimeout(r, POLL_MS));
      }
    } catch (e) {
      push({ role: "error", text: (e as Error).message }, false);
    } finally {
      polling = false;
      setTyping(false);
    }
  }

  async function sendText(): Promise<void> {
    const text = prepareOutgoing(input.value, config.max_chars);
    if (!text) return;
    input.value = "";
    push({ role: "user", text });
    send.disabled = true;
    try {
      const s = await ensureSession();
      await api("/web/messages", { method: "POST", body: JSON.stringify({ ...s, text }) });
      void poll();
    } catch (e) {
      if ((e as { status?: number }).status === 401) {
        state.session = undefined; // phiên hỏng -> xin phiên mới ở lần gửi sau
        store.save(state);
      }
      push({ role: "error", text: (e as Error).message }, false);
    } finally {
      send.disabled = false;
      input.focus();
    }
  }

  async function sendImage(f: File): Promise<void> {
    // ảnh gốc được phép lớn hơn giới hạn server vì sẽ được thu nhỏ trên trình duyệt trước khi gửi
    if (f.size > config.max_image_mb * 1_000_000 * 3) {
      push({ role: "error", text: `Ảnh quá lớn (tối đa khoảng ${config.max_image_mb} MB).` }, false);
      return;
    }
    const caption = prepareOutgoing(input.value, config.max_chars) ?? "";
    input.value = "";
    push({ role: "user", text: caption ? `🖼 ${f.name}\n${caption}` : `🖼 ${f.name}` });
    attach.disabled = send.disabled = true;
    try {
      const image_base64 = await downscaleToJpegBase64(f, 1600, 0.85);
      const s = await ensureSession();
      await api("/web/images", { method: "POST", body: JSON.stringify({ ...s, image_base64, caption }) });
      void poll();
    } catch (e) {
      push({ role: "error", text: (e as Error).message }, false);
    } finally {
      attach.disabled = send.disabled = false;
    }
  }

  fab.addEventListener("click", () => {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) {
      input.focus();
      if (state.session) void poll();
    }
  });
  close.addEventListener("click", () => panel.classList.remove("open"));
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    void sendText();
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      void sendText();
    }
  });
  attach.addEventListener("click", () => file.click());
  file.addEventListener("change", () => {
    const f = file.files?.[0];
    file.value = "";
    if (f) void sendImage(f);
  });

  for (const m of state.msgs) render(m);
  if (!state.msgs.length) render({ role: "bot", text: `Chào bạn! Mình là trợ lý AI tự động của ${TITLE}. Bạn cần hỗ trợ gì nè?` });
  api<WidgetConfig>("/web/config")
    .then((c) => {
      config = c;
      attach.hidden = !c.images;
    })
    .catch(() => { /* dùng cấu hình mặc định */ });
}

/** Thu nhỏ ảnh trên trình duyệt (cạnh dài tối đa maxSide) và mã hóa JPEG -> gửi ít dữ liệu, bỏ EXIF/GPS. */
async function downscaleToJpegBase64(f: File, maxSide: number, quality: number): Promise<string> {
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(f);
  } catch {
    throw new Error("Ảnh không đọc được. Bạn thử lại với ảnh JPEG, PNG hoặc WEBP khác nhé.");
  }
  const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
else mount();
