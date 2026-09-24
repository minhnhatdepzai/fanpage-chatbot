import { strict as assert } from "node:assert";
import { test } from "node:test";

import { isSafeHttpUrl, linkify, prepareOutgoing } from "../src/text.ts";

test("linkify tách link http(s) và bỏ dấu câu cuối", () => {
  const segs = linkify("Xem https://arxiv.org/abs/2106.09685. Hết");
  assert.deepEqual(segs, [
    { kind: "text", value: "Xem " },
    { kind: "link", value: "https://arxiv.org/abs/2106.09685", href: "https://arxiv.org/abs/2106.09685" },
    { kind: "text", value: ". Hết" },
  ]);
});

test("không tạo link cho javascript: hay chuỗi HTML", () => {
  const segs = linkify('<img src=x onerror=alert(1)> javascript:alert(1)');
  assert.ok(segs.every((s) => s.kind === "text"));
  assert.equal(isSafeHttpUrl("javascript:alert(1)"), false);
});

test("prepareOutgoing cắt khoảng trắng và chặn tin rỗng/quá dài", () => {
  assert.equal(prepareOutgoing("  chào \r\n bạn  ", 100), "chào \n bạn");
  assert.equal(prepareOutgoing("   ", 100), null);
  assert.equal(prepareOutgoing("x".repeat(101), 100), null);
});
