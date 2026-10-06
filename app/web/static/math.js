"use strict";

const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const form = $("#composer");
const input = $("#question");
const send = $("#send");
const imageInput = $("#image");
const literatureForm = $("#literature-form");
const literatureAnswer = $("#literature-answer");
const NS = "http://www.w3.org/2000/svg";
let chatSession = null;
let lastChatMessageId = 0;
let selectedLiteratureExam = null;
let lastLiteratureTopic = "";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function addUser(text) {
  const article = el("article", "message user");
  article.append(el("div", "bubble", text));
  conversation.append(article);
  scrollDown();
}

function addLoading() {
  const article = el("article", "message assistant");
  article.id = "math-loading";
  article.append(el("div", "avatar", "S"));
  const bubble = el("div", "bubble");
  const dots = el("div", "loading-dots");
  dots.append(el("i"), el("i"), el("i"));
  bubble.append(dots, el("p", "queue-status", "Đang xếp lượt xử lý…"));
  article.append(bubble);
  conversation.append(article);
  scrollDown();
}

function updateQueueStatus(queue) {
  const target = $("#math-loading .queue-status");
  if (!target || !queue) return;
  if (queue.phase === "processing") {
    target.textContent = `Đang xử lý câu hỏi của bạn · ${queue.active_jobs}/${queue.worker_slots || "?"} lượt đang chạy`;
  } else if (queue.phase === "queued") {
    target.textContent = `Bạn đang ở vị trí ${queue.position}/${queue.total} trong hàng chờ · ${queue.worker_slots || 0} luồng xử lý sẵn sàng`;
  } else target.textContent = "Đang hoàn tất câu trả lời…";
}

function scrollDown() {
  conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
}

function line(svg, x1, y1, x2, y2, attrs = {}) {
  svg.append(svgEl("line", { x1, y1, x2, y2, stroke: "#aaa4bd", "stroke-width": 2, ...attrs }));
}

function text(svg, x, y, value, attrs = {}) {
  const node = svgEl("text", { x, y, fill: "#514c62", "font-size": 13, "text-anchor": "middle", ...attrs });
  node.textContent = value;
  svg.append(node);
}

function niceStep(span, targetTicks = 7) {
  const rough = Math.max(span, 1e-9) / targetTicks;
  const power = 10 ** Math.floor(Math.log10(rough));
  const ratio = rough / power;
  const multiplier = ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10;
  return multiplier * power;
}

function axisRange(values) {
  let min = Math.min(0, ...values);
  let max = Math.max(0, ...values);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.1;
  return [min - pad, max + pad];
}

function shortNumber(value) {
  const clean = Math.abs(value) < 1e-9 ? 0 : value;
  return Math.abs(clean) >= 1000 ? clean.toExponential(1) : Number(clean.toFixed(2)).toString();
}

function drawVisual(data, steps = []) {
  const scene = window.STUDYSCOPE_SCENES?.create(data, steps);
  if (scene) return scene;
  const svg = svgEl("svg", { viewBox: "0 0 720 300", class: "math-visual", role: "img", "aria-label": "Minh họa trực quan chuyển động được dựng từ dữ liệu bài toán" });
  const type = data.type;
  if (type === "fraction") {
    const total = data.denominator;
    const width = 560 / total;
    for (let i = 0; i < total; i++) {
      svg.append(svgEl("rect", { x: 80 + i * width, y: 75, width, height: 105, rx: 5, fill: i < data.numerator ? "#bfe6cf" : "#fffef9", stroke: "#1e5a43", "stroke-width": 2, class: "visual-piece", style: `--delay:${(i * 90) + 250}ms` }));
    }
    text(svg, 360, 50, `${data.numerator}/${data.denominator}`, { "font-size": 25, "font-family": "Georgia", class: "visual-caption" });
  } else if (type === "number_line") {
    const end = data.start + data.change;
    const min = Math.min(0, data.start, end) - 1;
    const max = Math.max(data.start, end) + 1;
    const map = (v) => 55 + ((v - min) / (max - min)) * 610;
    line(svg, 50, 145, 675, 145, { stroke: "#1e5a43", "stroke-width": 3, class: "visual-axis" });
    for (let i = Math.ceil(min); i <= Math.floor(max); i++) { line(svg, map(i), 137, map(i), 153, { class: "visual-grid" }); text(svg, map(i), 174, i, { class: "visual-grid" }); }
    const startX = map(data.start), endX = map(end);
    svg.append(svgEl("path", { d: `M ${startX} 126 Q ${(startX + endX) / 2} 48 ${endX} 126`, pathLength: 1, fill: "none", stroke: "#ee8959", "stroke-width": 5, "stroke-linecap": "round", class: "visual-trace" }));
    svg.append(svgEl("circle", { cx: endX, cy: 126, r: 7, fill: "#ee8959", class: "visual-point", style: "--delay:1100ms" }));
    text(svg, (startX + endX) / 2, 69, `${data.change >= 0 ? "+" : ""}${data.change}`, { fill: "#a64721", "font-size": 18, "font-weight": 700, class: "visual-caption" });
  } else if (type === "groups") {
    const groups = Math.min(data.groups, 10), per = Math.min(data.per_group, 12);
    const gap = 620 / groups;
    for (let g = 0; g < groups; g++) {
      svg.append(svgEl("rect", { x: 50 + g * gap, y: 55, width: gap - 10, height: 140, rx: 14, fill: "#eff7ef", stroke: "#9ab4a5", class: "visual-piece", style: `--delay:${250 + g * 140}ms` }));
      for (let n = 0; n < per; n++) {
        const cols = Math.ceil(Math.sqrt(per));
        svg.append(svgEl("circle", { cx: 66 + g * gap + (n % cols) * 17, cy: 77 + Math.floor(n / cols) * 18, r: 6, fill: "#1e5a43", class: "visual-point", style: `--delay:${400 + g * 140 + n * 35}ms` }));
      }
    }
    text(svg, 360, 226, `${data.groups} nhóm × ${data.per_group} phần tử`, { "font-size": 16, class: "visual-caption" });
  } else if (type === "urn_probability") {
    svg.append(svgEl("path", { d: "M70 48 L94 238 Q205 286 316 238 L340 48 Z", pathLength: 1, fill: "#fbf9ff", stroke: "#7963cc", "stroke-width": 4, class: "visual-outline" }));
    let ballIndex = 0;
    (data.groups || []).forEach((group) => {
      for (let i = 0; i < Math.min(group.count, 12); i++) {
        const column = ballIndex % 6, row = Math.floor(ballIndex / 6);
        svg.append(svgEl("circle", { cx: 112 + column * 38, cy: 205 - row * 38, r: 13, fill: group.color, stroke: "#fff", "stroke-width": 2, class: "visual-point", style: `--delay:${260 + ballIndex * 70}ms` }));
        ballIndex += 1;
      }
    });
    const samples = (data.groups || []).slice(0, 2);
    samples.forEach((group, index) => {
      svg.append(svgEl("circle", { cx: 455 + index * 82, cy: 128, r: 24, fill: group.color, stroke: "#fff", "stroke-width": 4, class: "visual-move", style: `--delay:${1300 + index * 300}ms` }));
      text(svg, 455 + index * 82, 174, group.label, { "font-size": 13, class: "visual-caption", style: `--delay:${1600 + index * 300}ms` });
    });
    text(svg, 538, 67, `Lấy ${data.draw} viên không hoàn lại`, { "font-size": 16, "font-weight": 700, class: "visual-caption" });
    text(svg, 538, 223, `${data.favorable}/${data.total} = ${data.probability}`, { fill: "#5f4bb4", "font-size": 22, "font-weight": 800, class: "visual-result" });
  } else if (type === "arithmetic_sequence") {
    const rows = data.rows || [];
    const maxSeats = Math.max(1, ...rows.map((row) => row.seats));
    rows.forEach((row, index) => {
      const y = 43 + index * 45;
      const width = 115 + (row.seats / maxSeats) * 390;
      text(svg, 92, y + 8, `Hàng ${row.row}`, { "font-size": 13, "font-weight": 700, "text-anchor": "end", class: "visual-caption", style: `--delay:${180 + index * 230}ms` });
      svg.append(svgEl("rect", { x: 112, y: y - 13, width, height: 28, rx: 8, fill: index === rows.length - 1 ? "#7963cc" : "#d9d2f4", class: "visual-grow", style: `--delay:${260 + index * 230}ms` }));
      svg.append(svgEl("circle", { cx: 130, cy: y - 3, r: 5, fill: "#fff", class: "visual-point", style: `--delay:${420 + index * 230}ms` }));
      line(svg, 130, y + 2, 130, y + 11, { stroke: "#fff", "stroke-width": 3, "stroke-linecap": "round", class: "visual-piece", style: `--delay:${420 + index * 230}ms` });
      text(svg, 112 + width - 10, y + 6, `${row.seats} ghế`, { fill: index === rows.length - 1 ? "#fff" : "#514c62", "font-size": 13, "font-weight": 800, "text-anchor": "end", class: "visual-caption", style: `--delay:${500 + index * 230}ms` });
    });
    text(svg, 680, 24, `Mỗi hàng ${data.difference >= 0 ? "+" : ""}${data.difference} ghế`, { fill: "#6756b7", "font-size": 14, "font-weight": 800, "text-anchor": "end", class: "visual-caption" });
    text(svg, 680, 278, `S${data.count} = ${data.total} ghế`, { fill: "#17747c", "font-size": 20, "font-weight": 900, "text-anchor": "end", class: "visual-result" });
  } else if (type === "rate_grid") {
    const drawPeople = (count, y, color, delay) => {
      for (let i = 0; i < Math.min(count, 10); i++) {
        const x = 78 + i * 42;
        svg.append(svgEl("circle", { cx: x, cy: y, r: 10, fill: color, class: "visual-point", style: `--delay:${delay + i * 80}ms` }));
        line(svg, x, y + 10, x, y + 32, { stroke: color, "stroke-width": 5, "stroke-linecap": "round", class: "visual-piece", style: `--delay:${delay + i * 80}ms` });
      }
    };
    drawPeople(data.base_people, 72, "#7963cc", 250);
    drawPeople(data.target_people, 186, "#2698a2", 950);
    text(svg, 620, 78, `${data.base_people} HS × ${data.base_hours} giờ → ${data.base_output} thiệp`, { "font-size": 15, "text-anchor": "end", class: "visual-caption" });
    text(svg, 620, 192, `${data.target_people} HS × ${data.target_hours} giờ → ${data.target_output} thiệp`, { fill: "#17747c", "font-size": 16, "font-weight": 800, "text-anchor": "end", class: "visual-result" });
    svg.append(svgEl("path", { d: "M360 112 L360 149", pathLength: 1, stroke: "#ee8959", "stroke-width": 6, "stroke-linecap": "round", class: "visual-trace" }));
    svg.append(svgEl("path", { d: "M348 140 L360 153 L372 140", fill: "none", stroke: "#ee8959", "stroke-width": 5, class: "visual-trace" }));
  } else if (type === "rectangle") {
    const scale = Math.min(450 / data.length, 150 / data.width);
    const width = scale * data.length;
    const height = scale * data.width;
    const x = 360 - width / 2, y = 120 - height / 2;
    svg.append(svgEl("rect", { x, y, width, height, rx: 5, pathLength: 1, fill: "#bfe6cf", stroke: "#1e5a43", "stroke-width": 4, class: "visual-outline" }));
    text(svg, 360, y - 14, `dài ${data.length}`, { "font-size": 15, class: "visual-caption" });
    text(svg, x - 38, 124, `rộng ${data.width}`, { "font-size": 15, transform: `rotate(-90 ${x - 38} 124)`, class: "visual-caption" });
    text(svg, 360, 230, `Kết quả: ${data.result}`, { fill: "#1e5a43", "font-size": 19, "font-weight": 700, class: "visual-result" });
  } else if (type === "balance") {
    const balance = svgEl("g", { class: "visual-balance" });
    line(balance, 360, 65, 360, 200, { stroke: "#1e5a43", "stroke-width": 7 });
    line(balance, 190, 100, 530, 100, { stroke: "#1e5a43", "stroke-width": 5 });
    line(balance, 220, 100, 185, 160); line(balance, 500, 100, 535, 160);
    balance.append(svgEl("path", { d: "M125 160 Q185 215 245 160 Z", fill: "#bfe6cf", stroke: "#1e5a43" }));
    balance.append(svgEl("path", { d: "M475 160 Q535 215 595 160 Z", fill: "#d9ef86", stroke: "#1e5a43" }));
    text(balance, 185, 153, data.left, { "font-size": 19, "font-weight": 700 });
    text(balance, 535, 153, data.right, { "font-size": 19, "font-weight": 700 });
    svg.append(balance);
    text(svg, 360, 232, data.solution, { fill: "#1e5a43", "font-size": 19, "font-weight": 700, class: "visual-result" });
  } else if (type === "graph" || type === "coordinate_system" || type === "coordinate_segment") {
    const sets = (type === "graph" || type === "coordinate_segment" ? [data.points] : data.lines || [])
      .map((points) => (points || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
      .filter((points) => points.length > 1);
    const markers = (data.markers || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const allPoints = sets.flat().concat(markers);
    if (!allPoints.length) return window.STUDYSCOPE_SCENES.create({ type: "formula" }, steps);
    let [xMin, xMax] = axisRange(allPoints.map((p) => p.x));
    let [yMin, yMax] = axisRange(allPoints.map((p) => p.y));
    const left = 64, right = 695, top = 34, bottom = 264;
    if (type === "coordinate_segment") {
      const scale = Math.min((right - left) / (xMax - xMin), (bottom - top) / (yMax - yMin));
      const centerX = (xMin + xMax) / 2, centerY = (yMin + yMax) / 2;
      const halfX = (right - left) / (2 * scale), halfY = (bottom - top) / (2 * scale);
      xMin = centerX - halfX; xMax = centerX + halfX;
      yMin = centerY - halfY; yMax = centerY + halfY;
    }
    const mapX = (value) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
    const mapY = (value) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);
    const xStep = niceStep(xMax - xMin);
    const yStep = niceStep(yMax - yMin);
    const xTicks = [];
    const yTicks = [];
    for (let value = Math.ceil(xMin / xStep) * xStep; value <= xMax + xStep / 10 && xTicks.length < 14; value += xStep) xTicks.push(value);
    for (let value = Math.ceil(yMin / yStep) * yStep; value <= yMax + yStep / 10 && yTicks.length < 14; value += yStep) yTicks.push(value);
    xTicks.forEach((value) => {
      const x = mapX(value); line(svg, x, top, x, bottom, { stroke: "#ebe8f2", "stroke-width": 1, class: "visual-grid" });
      text(svg, x, bottom + 18, shortNumber(value), { fill: "#8d879d", "font-size": 10, class: "visual-grid" });
    });
    yTicks.forEach((value) => {
      const y = mapY(value); line(svg, left, y, right, y, { stroke: "#ebe8f2", "stroke-width": 1, class: "visual-grid" });
      if (Math.abs(value) > 1e-9) text(svg, left - 9, y + 4, shortNumber(value), { fill: "#8d879d", "font-size": 10, "text-anchor": "end", class: "visual-grid" });
    });
    const axisX = mapY(0), axisY = mapX(0);
    line(svg, left, axisX, right, axisX, { stroke: "#6f687e", "stroke-width": 1.8, class: "visual-axis" });
    line(svg, axisY, top, axisY, bottom, { stroke: "#6f687e", "stroke-width": 1.8, class: "visual-axis" });
    text(svg, right - 4, axisX - 8, data.variable || "x", { fill: "#5f586f", "font-size": 12, "font-style": "italic", "text-anchor": "end", class: "visual-axis" });
    text(svg, axisY + 12, top + 10, "y", { fill: "#5f586f", "font-size": 12, "font-style": "italic", "text-anchor": "start", class: "visual-axis" });
    const colors = ["#7963cc", "#ef8a6c", "#2698a2"];
    sets.forEach((points, index) => {
      const path = points.map((p, i) => `${i ? "L" : "M"} ${mapX(p.x)} ${mapY(p.y)}`).join(" ");
      svg.append(svgEl("path", { d: path, pathLength: 1, fill: "none", stroke: colors[index % colors.length], "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round", class: "visual-trace", style: `--delay:${650 + index * 250}ms` }));
      points.filter((point, pointIndex) => type !== "graph" || pointIndex % 10 === 0).forEach((point, pointIndex) => svg.append(svgEl("circle", { cx: mapX(point.x), cy: mapY(point.y), r: 3.2, fill: "#fff", stroke: colors[index % colors.length], "stroke-width": 2, class: "visual-point", style: `--delay:${1100 + index * 250 + pointIndex * 55}ms` })));
      points.filter((point) => point.label).forEach((point, pointIndex) => text(svg, mapX(point.x) + 10, mapY(point.y) - 10, point.label, { fill: colors[index % colors.length], "font-size": 14, "font-weight": 800, "text-anchor": "start", class: "visual-caption", style: `--delay:${1300 + pointIndex * 180}ms` }));
    });
    markers.forEach((point) => {
      svg.append(svgEl("circle", { cx: mapX(point.x), cy: mapY(point.y), r: 5, fill: "#ef8a6c", stroke: "#fff", "stroke-width": 2, class: "visual-point graph-marker" }));
      const nearRight = mapX(point.x) > right - 150;
      text(svg, mapX(point.x) + (nearRight ? -8 : 8), Math.max(top + 12, mapY(point.y) - 12), point.label || `(${shortNumber(point.x)}; ${shortNumber(point.y)})`, { fill: "#5f4bb4", "font-size": 12, "text-anchor": nearRight ? "end" : "start", class: "visual-caption graph-marker-label" });
    });
    if (data.expression) text(svg, right, 20, `y = ${data.expression.replaceAll("*", "·")}`, { fill: "#6756b7", "font-size": 13, "font-weight": 700, "text-anchor": "end", class: "visual-caption" });
    if (data.solution) text(svg, right, 20, data.solution, { fill: "#6756b7", "font-size": 12, "font-weight": 700, "text-anchor": "end", class: "visual-caption" });
  } else {
    if (!data.expression || data.result === undefined) {
      const fallback = window.STUDYSCOPE_SCENES?.create({ type: "formula" }, steps);
      if (fallback) return fallback;
    }
    svg.append(svgEl("rect", { x: 60, y: 56, width: 600, height: 135, rx: 20, pathLength: 1, fill: "#f0edff", stroke: "#9f91dc", class: "visual-outline" }));
    text(svg, 360, 113, data.expression || "Biểu thức", { "font-size": 25, "font-family": "Georgia", class: "visual-caption" });
    text(svg, 360, 157, `= ${data.result || ""}`, { fill: "#5f4bb4", "font-size": 28, "font-weight": 700, class: "visual-result" });
  }
  const stepNodes = [...svg.querySelectorAll(".visual-trace,.visual-point,.visual-result")];
  svg.updateLessonStep = (index) => stepNodes.forEach((n) => {
    const revealAt = n.classList.contains("visual-result") ? Math.max(1, steps.length - 1) : n.classList.contains("visual-point") ? Math.min(2, steps.length - 1) : 1;
    n.style.visibility = index >= revealAt ? "visible" : "hidden";
  });
  return svg;
}

function downloadSvg(svg) {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", NS);
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = "mathscope-visual.svg"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function setupSolutionAnimation(stage, svg, stepItems, pauseButton, replayButton) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const state = { index: -1, timer: null, paused: false, finished: false };
  const clearTimer = () => { if (state.timer) window.clearTimeout(state.timer); state.timer = null; };
  const setStepState = () => {
    stepItems.forEach((item, index) => {
      item.classList.toggle("is-revealed", index <= state.index);
      item.classList.toggle("is-active", index === state.index && !state.finished);
      item.classList.toggle("is-complete", index < state.index || (state.finished && index <= state.index));
      const status = item.querySelector(".step-state");
      if (status) status.textContent = index === state.index && !state.finished ? "Đang minh họa" : index <= state.index ? "Đã xong" : "Chờ";
    });
    stage.style.setProperty("--step-progress", String(Math.max(0, state.index + 1) / Math.max(1, stepItems.length)));
    svg.updateLessonStep?.(Math.max(0, state.index));
    const caption = stage.querySelector(".lesson-step-caption");
    if (caption && state.index >= 0) caption.textContent = `Bước ${state.index + 1} · ${stepItems[state.index].querySelector(".step-copy p").textContent}`;
  };
  const finish = () => {
    state.finished = true;
    state.index = stepItems.length - 1;
    setStepState();
    pauseButton.disabled = true;
    pauseButton.textContent = "Đã xong";
    replayButton.textContent = "↻ Phát lại";
  };
  const advance = () => {
    if (state.paused || state.finished) return;
    if (state.index >= stepItems.length - 1) { finish(); return; }
    state.index += 1;
    setStepState();
    if (state.index >= stepItems.length - 1) state.timer = window.setTimeout(finish, 2600);
    else state.timer = window.setTimeout(advance, 3000);
  };
  const play = () => {
    clearTimer();
    state.index = -1; state.paused = false; state.finished = false;
    stage.classList.remove("is-paused", "is-seeking");
    svg.classList.remove("is-playing");
    void svg.getBoundingClientRect();
    svg.classList.add("is-playing");
    svg.setCurrentTime?.(0); svg.unpauseAnimations?.();
    pauseButton.disabled = false;
    pauseButton.textContent = "Ⅱ Tạm dừng";
    replayButton.textContent = "↻ Xem lại";
    setStepState();
    if (reduceMotion) {
      finish();
      svg.pauseAnimations?.();
    } else {
      state.timer = window.setTimeout(advance, 180);
    }
  };
  pauseButton.addEventListener("click", () => {
    if (state.finished) return;
    state.paused = !state.paused;
    stage.classList.toggle("is-paused", state.paused);
    pauseButton.textContent = state.paused ? "▶ Tiếp tục" : "Ⅱ Tạm dừng";
    if (state.paused) { clearTimer(); svg.pauseAnimations?.(); }
    else { svg.unpauseAnimations?.(); state.timer = window.setTimeout(advance, 480); }
  });
  replayButton.addEventListener("click", play);
  stepItems.forEach((item, index) => item.addEventListener("click", () => {
    clearTimer();
    state.index = index; state.paused = true; state.finished = false;
    stage.classList.add("is-paused", "is-seeking");
    svg.pauseAnimations?.();
    pauseButton.disabled = false; pauseButton.textContent = "▶ Tiếp tục";
    setStepState();
  }));
  window.requestAnimationFrame(play);
}

function addSolution(result) {
  if (result.status !== "reviewed" && (result.status !== "verified" || result.verification?.passed !== true)) {
    addError("Kết quả chưa vượt qua kiểm chứng.", "Chưa thể chốt đáp số cho đề này.");
    return;
  }
  const article = el("article", "message assistant");
  article.append(el("div", "avatar", "S"));
  const bubble = el("div", "bubble");
  const head = el("div", "answer-head");
  const answer = el("div");
  answer.append(el("p", "message-label", result.status === "reviewed" ? "LỜI GIẢI THAM KHẢO" : "LỜI GIẢI VÀ ĐỐI CHIẾU"), el("div", "answer-value", result.answer));
  const verifiedLabel = result.verification.method === "exact_symbolic_sampling"
    ? "✓ ĐIỂM TÍNH CHÍNH XÁC"
    : result.status === "reviewed" ? "ĐỐI CHIẾU BẰNG AI" : "✓ ĐÃ KIỂM TRA KẾT QUẢ";
  head.append(answer, el("span", "badge", verifiedLabel));
  bubble.append(head, el("div", "meta", `${result.topic}${result.grade ? ` • Lớp ${result.grade}` : ""}`));
  if (result.image_analysis) {
    const recognized = el("div", "recognized");
    recognized.append(el("strong", "", "Đề nhận dạng từ ảnh"), el("p", "", result.image_analysis.extracted_text));
    bubble.append(recognized);
  }
  const stage = el("div", "solution-stage");
  const stepsPanel = el("section", "solution-steps-panel");
  const stepsHeader = el("div", "steps-header");
  const stepsHeading = el("div");
  stepsHeading.append(el("span", "", "TRÌNH TỰ GIẢI"), el("strong", "", `${result.steps.length} bước`));
  const progressTrack = el("span", "step-progress-track");
  progressTrack.append(el("i"));
  stepsHeader.append(stepsHeading, progressTrack);
  const steps = el("ol", "steps");
  const stepItems = [];
  result.steps.forEach((step, index) => {
    const item = el("li", "step-item");
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-label", `Xem bước ${index + 1}: ${step.title}`);
    item.append(el("span", "step-num", String(index + 1)));
    const content = el("div", "step-copy");
    const titleRow = el("div", "step-title-row");
    titleRow.append(el("strong", "", step.title), el("span", "step-state", "Chờ"));
    content.append(titleRow, el("p", "", step.detail));
    item.append(content); steps.append(item);
    item.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); item.click(); } });
    stepItems.push(item);
  });
  stepsPanel.append(stepsHeader, steps);
  const visualCard = el("div", "visual-card");
  const toolbar = el("div", "visual-toolbar");
  const visualTitle = el("div");
  visualTitle.append(el("strong", "", ["graph", "coordinate_system", "coordinate_segment"].includes(result.visual.type) ? "ĐỒ THỊ TỌA ĐỘ" : "HÌNH MINH HỌA"));
  visualTitle.append(el("span", "", result.status === "reviewed" ? "Sơ đồ trình tự lập luận — chưa phải kiểm chứng ký hiệu" : "Dựng tự động từ dữ liệu đã kiểm chứng"));
  toolbar.append(visualTitle);
  const visualActions = el("div", "visual-actions");
  const pause = el("button", "animation-control pause-animation", "Ⅱ Tạm dừng"); pause.type = "button";
  const replay = el("button", "animation-control replay-animation", "↻ Xem lại"); replay.type = "button";
  const expand = el("button", "animation-control expand-animation", "⛶ Phóng lớn"); expand.type = "button";
  expand.addEventListener("click", () => stage.requestFullscreen?.().catch(() => {}));
  const download = el("button", "download", "Tải SVG ↓"); download.type = "button";
  const svg = drawVisual(result.visual, result.steps); download.addEventListener("click", () => downloadSvg(svg));
  visualActions.append(pause, replay, expand, download);
  toolbar.append(visualActions); visualCard.append(toolbar, svg);
  visualCard.append(el("p", "lesson-step-caption", "Bấm vào một bước bên trái để xem lại hình tương ứng."));
  if (result.visual.type === "arithmetic_sequence") {
    const controls = el("label", "row-scrubber", "Khám phá từng hàng ghế ");
    const range = el("input"); range.type = "range"; range.min = "1"; range.max = String(Math.min(Math.max(result.visual.nth, result.visual.count), 24)); range.value = "1";
    range.setAttribute("aria-label", "Chọn hàng ghế để quan sát");
    range.addEventListener("input", () => svg.updateLessonStep?.(result.steps.length - 1, Number(range.value)));
    controls.append(range); visualCard.append(controls);
  }
  if (result.visual.points?.length) {
    const pointList = el("div", "visual-points");
    result.visual.points.slice(0, 11).forEach((point) => pointList.append(el("span", "", `(${shortNumber(point.x)}; ${shortNumber(point.y)})`)));
    visualCard.append(pointList);
  }
  stage.append(stepsPanel, visualCard);
  bubble.append(stage);
  if (result.sources && result.sources.length) {
    const source = el("div", "source", "Khung chương trình tham chiếu: ");
    const link = el("a", "", result.sources[0].title); link.href = result.sources[0].url; link.target = "_blank"; link.rel = "noopener noreferrer";
    source.append(link); bubble.append(source);
  }
  (result.warnings || []).forEach((warning) => bubble.append(el("p", "warning", `Lưu ý: ${warning}`)));
  article.append(bubble); conversation.append(article); recordCompleted(); scrollDown();
  setupSolutionAnimation(stage, svg, stepItems, pause, replay);
}

function addError(message, hint) {
  const article = el("article", "message assistant error");
  article.append(el("div", "avatar", "S"));
  const bubble = el("div", "bubble");
  bubble.append(el("p", "message-label", "CHƯA THỂ KIỂM CHỨNG"), el("strong", "", message));
  if (hint) bubble.append(el("p", "", hint));
  article.append(bubble); conversation.append(article); scrollDown();
}

function addMathReview(question, response) {
  const cleaned = response.replace(/\\n/g, "\n").split(/\n\n\(Lưu ý:/, 1)[0];
  const parts = cleaned.split(/\n\s*\n|(?=Bước\s+\d+[:.])/).map((part) => part.trim()).filter(Boolean);
  const steps = parts.map((detail, index) => ({ title: `Phần ${index + 1}`, detail }));
  if (!steps.length) steps.push({ title: "Đọc đề", detail: question });
  const refused = /chưa thể kiểm chứng|chưa thể chứng minh|thiếu (?:dữ kiện|giả thiết)/i.test(cleaned);
  addSolution({status: "reviewed", answer: refused ? "Cần làm rõ dữ kiện" : "Lời giải và kiểm tra", topic: "Phân tích bài toán", grade: null,
    steps, visual: {type: "formula", expression: question, result: cleaned.slice(0, 200)},
    verification: {method: "ai_review", engine: "ai"}, sources: [], warnings: []});
}

const ttsVoices = [
  { id: "vi-VN-HoaiMyNeural", label: "Hoài My · Việt Nam", locale: "vi-VN" },
  { id: "vi-VN-NamMinhNeural", label: "Nam Minh · Việt Nam", locale: "vi-VN" },
  { id: "en-US-JennyNeural", label: "Jenny · Mỹ", locale: "en-US" },
  { id: "en-US-GuyNeural", label: "Guy · Mỹ", locale: "en-US" },
  { id: "en-GB-SoniaNeural", label: "Sonia · Anh", locale: "en-GB" },
  { id: "en-GB-RyanNeural", label: "Ryan · Anh", locale: "en-GB" },
];

const ttsStyles = [
  { id: "teacher", label: "Giáo viên", rate: -5, pitch: 0 },
  { id: "podcast", label: "Podcast", rate: -8, pitch: -2 },
  { id: "story", label: "Kể chuyện", rate: -12, pitch: 3 },
  { id: "clear", label: "Chậm & rõ", rate: -20, pitch: 0 },
];

function cleanNarrationText(value) {
  return value
    .split(/\n\nNguồn:\s*\n/i, 1)[0]
    .replace(/\[([^\]]+)]\(https?:\/\/[^)]+\)/g, "$1")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/[`*_#>~|]/g, " ")
    .replace(/\s*\n\s*/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim()
    .slice(0, 24000);
}

function narrationSegments(value) {
  const clean = cleanNarrationText(value);
  const segments = clean.match(/[^.!?…\n]+(?:[.!?…]+|\n|$)/g)?.map((part) => part.trim()).filter(Boolean) || [];
  return segments.length ? segments : [clean];
}

function formatAudioTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  return `${mins}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function buildPodcastPlayer(answerText, mode) {
  const narration = cleanNarrationText(answerText);
  const segments = narrationSegments(answerText);
  const card = el("section", `podcast-player ${mode === "english" ? "english-podcast" : "literature-podcast"}`);
  const visual = el("div", "podcast-visual");
  const speaker = el("div", "podcast-speaker", mode === "english" ? "Aa" : "文");
  const wave = el("div", "podcast-wave");
  for (let index = 0; index < 18; index++) wave.append(el("i"));
  const visualCopy = el("div", "podcast-visual-copy");
  visualCopy.append(
    el("span", "podcast-kicker", mode === "english" ? "ENGLISH AUDIO LESSON" : "NGỮ VĂN AUDIO LESSON"),
    el("strong", "", mode === "english" ? "Learn by listening" : "Nghe để hiểu sâu hơn"),
  );
  const subtitle = el("p", "podcast-subtitle", segments[0] || "Sẵn sàng phát bài đọc.");
  visual.append(speaker, visualCopy, wave, subtitle);

  const controls = el("div", "podcast-controls");
  const settingsRow = el("div", "podcast-settings");
  const voiceLabel = el("label"); voiceLabel.append(el("span", "", "Giọng đọc"));
  const voiceSelect = el("select", "voice-select");
  ttsVoices.forEach((voice) => {
    const option = el("option", "", voice.label); option.value = voice.id; voiceSelect.append(option);
  });
  voiceSelect.value = mode === "english" ? "en-US-JennyNeural" : "vi-VN-HoaiMyNeural";
  voiceLabel.append(voiceSelect);
  const styleLabel = el("label"); styleLabel.append(el("span", "", "Nhấn nhá"));
  const styleSelect = el("select", "style-select");
  ttsStyles.forEach((style) => { const option = el("option", "", style.label); option.value = style.id; styleSelect.append(option); });
  styleSelect.value = "podcast"; styleLabel.append(styleSelect);
  settingsRow.append(voiceLabel, styleLabel);

  const sliders = el("div", "podcast-sliders");
  const slider = (labelText, className, min, max, value, unit) => {
    const label = el("label");
    const head = el("span"); head.append(el("b", "", labelText), el("em", "", `${value}${unit}`));
    const inputNode = el("input", className); inputNode.type = "range"; inputNode.min = min; inputNode.max = max; inputNode.value = value;
    inputNode.addEventListener("input", () => { head.querySelector("em").textContent = `${Number(inputNode.value) > 0 ? "+" : ""}${inputNode.value}${unit}`; });
    label.append(head, inputNode); return { label, input: inputNode };
  };
  const rate = slider("Tốc độ", "tts-rate", -30, 30, 0, "%");
  const pitch = slider("Cao độ", "tts-pitch", -20, 20, 0, "Hz");
  const volume = slider("Âm lượng", "tts-volume", -30, 20, 0, "%");
  sliders.append(rate.label, pitch.label, volume.label);

  const transport = el("div", "podcast-transport");
  const play = el("button", "podcast-play", "▶ Nghe bài"); play.type = "button";
  const stop = el("button", "podcast-stop", "■"); stop.type = "button"; stop.title = "Dừng bài đọc";
  const seek = el("input", "podcast-seek"); seek.type = "range"; seek.min = "0"; seek.max = "1000"; seek.value = "0";
  const time = el("span", "podcast-time", "0:00 / 0:00");
  const download = el("a", "podcast-download", "Tải MP3"); download.hidden = true; download.download = mode === "english" ? "english-lesson.mp3" : "ngu-van-podcast.mp3";
  transport.append(play, stop, seek, time, download);
  const status = el("p", "podcast-status", "Giọng neural · MP3 không được lưu trên máy chủ");
  controls.append(settingsRow, sliders, transport, status);
  card.append(visual, controls);

  const audio = new Audio();
  let objectUrl = "";
  let fallbackUtterance = null;
  let usingFallback = false;
  let loadedKey = "";
  const currentKey = () => [voiceSelect.value, styleSelect.value, rate.input.value, pitch.input.value, volume.input.value].join("|");
  const setPlaying = (playing) => { card.classList.toggle("is-playing", playing); play.textContent = playing ? "Ⅱ Tạm dừng" : "▶ Tiếp tục"; };
  const setSubtitleByRatio = (ratio) => {
    const totalChars = Math.max(1, narration.length);
    const target = Math.max(0, Math.min(totalChars - 1, ratio * totalChars));
    let chars = 0; let selected = segments[segments.length - 1] || narration;
    for (const segment of segments) { chars += segment.length; if (target <= chars) { selected = segment; break; } }
    subtitle.textContent = selected;
  };
  const clearAudio = () => {
    audio.pause(); audio.removeAttribute("src"); audio.load();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = ""; loadedKey = ""; download.hidden = true;
  };
  const cancelFallback = () => { if (fallbackUtterance) window.speechSynthesis?.cancel(); fallbackUtterance = null; };
  const stopAll = () => { clearAudio(); cancelFallback(); usingFallback = false; seek.value = "0"; time.textContent = "0:00 / 0:00"; subtitle.textContent = segments[0] || narration; setPlaying(false); play.textContent = "▶ Nghe bài"; };
  const browserFallback = () => {
    if (!("speechSynthesis" in window)) throw new Error("Trình duyệt không có giọng đọc dự phòng.");
    usingFallback = true;
    const utterance = new SpeechSynthesisUtterance(narration);
    fallbackUtterance = utterance;
    const selectedVoice = ttsVoices.find((voice) => voice.id === voiceSelect.value);
    const browserVoice = window.speechSynthesis.getVoices().find((voice) => voice.lang.toLowerCase() === selectedVoice?.locale.toLowerCase())
      || window.speechSynthesis.getVoices().find((voice) => voice.lang.toLowerCase().startsWith((selectedVoice?.locale || "vi").slice(0, 2).toLowerCase()));
    if (browserVoice) utterance.voice = browserVoice;
    const style = ttsStyles.find((item) => item.id === styleSelect.value) || ttsStyles[1];
    utterance.lang = selectedVoice?.locale || "vi-VN";
    utterance.rate = Math.max(.55, Math.min(1.6, 1 + (style.rate + Number(rate.input.value)) / 100));
    utterance.pitch = Math.max(.5, Math.min(1.5, 1 + (style.pitch + Number(pitch.input.value)) / 50));
    utterance.volume = Math.max(.2, Math.min(1, 1 + Number(volume.input.value) / 100));
    utterance.onboundary = (event) => { if (event.charIndex >= 0) { const ratio = event.charIndex / Math.max(1, narration.length); seek.value = String(Math.round(ratio * 1000)); setSubtitleByRatio(ratio); } };
    utterance.onend = () => { setPlaying(false); play.textContent = "↻ Nghe lại"; seek.value = "1000"; setSubtitleByRatio(1); };
    utterance.onerror = () => { setPlaying(false); status.textContent = "Không phát được giọng đọc trên trình duyệt này."; };
    status.textContent = "Đang dùng giọng hệ thống dự phòng của trình duyệt.";
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(utterance); setPlaying(true);
  };
  const loadNeural = async () => {
    const session = await ensureChatSession();
    const response = await fetch("/web/tts/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...session,
        text: narration,
        voice: voiceSelect.value,
        style: styleSelect.value,
        rate: Number(rate.input.value),
        pitch: Number(pitch.input.value),
        volume: Number(volume.input.value),
      }),
    });
    if (!response.ok) throw new Error("Neural TTS unavailable");
    clearAudio();
    objectUrl = URL.createObjectURL(await response.blob());
    audio.src = objectUrl; loadedKey = currentKey();
    download.href = objectUrl; download.hidden = false;
    status.textContent = `Giọng neural · ${ttsVoices.find((voice) => voice.id === voiceSelect.value)?.label || voiceSelect.value}`;
    await audio.play(); usingFallback = false; setPlaying(true);
  };
  play.addEventListener("click", async () => {
    if (usingFallback && fallbackUtterance) {
      if (window.speechSynthesis.paused) { window.speechSynthesis.resume(); setPlaying(true); }
      else if (window.speechSynthesis.speaking) { window.speechSynthesis.pause(); setPlaying(false); }
      else browserFallback();
      return;
    }
    if (audio.src && loadedKey === currentKey()) {
      if (audio.paused) { await audio.play(); setPlaying(true); } else { audio.pause(); setPlaying(false); }
      return;
    }
    play.disabled = true; play.textContent = "Đang tạo giọng…"; status.textContent = "Đang dựng bài đọc neural, vui lòng chờ…";
    try { await loadNeural(); }
    catch (_) { try { browserFallback(); } catch (error) { status.textContent = error.message; } }
    finally { play.disabled = false; }
  });
  stop.addEventListener("click", stopAll);
  audio.addEventListener("timeupdate", () => {
    const ratio = audio.duration ? audio.currentTime / audio.duration : 0;
    seek.value = String(Math.round(ratio * 1000)); setSubtitleByRatio(ratio);
    time.textContent = `${formatAudioTime(audio.currentTime)} / ${formatAudioTime(audio.duration)}`;
  });
  audio.addEventListener("ended", () => { setPlaying(false); play.textContent = "↻ Nghe lại"; seek.value = "1000"; setSubtitleByRatio(1); });
  seek.addEventListener("input", () => { if (audio.duration && !usingFallback) audio.currentTime = (Number(seek.value) / 1000) * audio.duration; });
  [voiceSelect, styleSelect, rate.input, pitch.input, volume.input].forEach((control) => control.addEventListener("change", stopAll));
  return card;
}

function addChatAnswer(textValue, notice = "", mode = "chat") {
  const article = el("article", "message assistant");
  article.append(el("div", "avatar", "S"));
  const bubble = el("div", "bubble");
  const labels = { english: "GIA SƯ TIẾNG ANH", writing: "GIA SƯ NGỮ VĂN", math_review: "TOÁN • ĐÃ KIỂM TRA ĐỘC LẬP", chat: "TRỢ LÝ ĐA LĨNH VỰC" };
  bubble.append(el("p", "message-label", labels[mode] || labels.chat));
  if (notice) bubble.append(el("p", "route-note", notice));
  if (mode === "english" || mode === "writing") {
    bubble.append(buildPodcastPlayer(textValue, mode));
    const writtenAnswer = el("section", "written-answer");
    writtenAnswer.append(
      el("span", "written-answer-kicker", mode === "english" ? "FULL WRITTEN ANSWER" : "BÀI TRẢ LỜI ĐẦY ĐỦ"),
      el("h3", "", mode === "english" ? "Answer & explanation" : "Nội dung phân tích"),
      el("p", "chat-text", textValue),
    );
    bubble.append(writtenAnswer);
  } else {
    bubble.append(el("p", "chat-text", textValue));
  }
  if (route === "literature") {
    const longAction = el("button", "long-answer-cta");
    longAction.type = "button";
    longAction.append(el("span", "", "✦"), el("strong", "", "Viết thành bài hoàn chỉnh 2.500+ chữ"), el("b", "", "→"));
    longAction.addEventListener("click", () => requestLongLiteratureAnswer());
    bubble.append(longAction);
  }
  article.append(bubble); conversation.append(article); recordCompleted(); scrollDown();
}

async function jsonRequest(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  let body = {};
  try { body = await response.json(); } catch (_) { /* body remains empty */ }
  if (!response.ok) {
    const detail = body.detail || {};
    const message = typeof detail === "string" ? detail : detail.message;
    throw new Error(message || `Máy chủ trả lỗi ${response.status}.`);
  }
  return body;
}

async function ensureChatSession() {
  if (!chatSession) chatSession = await jsonRequest("/web/session", { method: "POST", body: "{}" });
  return chatSession;
}

async function waitForChatReply() {
  const session = await ensureChatSession();
  // Bài Văn/NLXH đa nguồn còn qua một lượt kiểm định độc lập nên có thể lâu hơn chat thường.
  const deadline = Date.now() + 600000;
  const received = [];
  let replyMode = "chat";
  while (Date.now() < deadline) {
    const params = new URLSearchParams({ session_id: session.session_id, token: session.token, after: String(lastChatMessageId) });
    const state = await jsonRequest(`/web/messages?${params}`);
    updateQueueStatus(state.queue);
    for (const message of state.messages || []) {
      lastChatMessageId = Math.max(lastChatMessageId, message.id);
      received.push(message.text);
      if (message.mode) replyMode = message.mode;
    }
    if (!state.pending && received.length) return { text: received.join("\n\n"), mode: replyMode };
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
  throw new Error("Chatbot phản hồi quá lâu. Bạn thử lại sau nhé.");
}

async function askGeneralChat(question, imageBase64 = "") {
  const session = await ensureChatSession();
  if (imageBase64) {
    const accepted = await jsonRequest("/web/images", { method: "POST", body: JSON.stringify({ ...session, image_base64: imageBase64, caption: question }) });
    updateQueueStatus(accepted.queue);
  } else {
    const accepted = await jsonRequest("/web/messages", { method: "POST", body: JSON.stringify({ ...session, text: question }) });
    updateQueueStatus(accepted.queue);
  }
  return waitForChatReply();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  const file = imageInput.files[0];
  if (!question && !file) return;
  if (route === "literature" && !question.startsWith("Hãy viết ngay một bài văn hoàn chỉnh khoảng 2800 chữ")) {
    lastLiteratureTopic = question;
  }
  addUser(file ? `${question || "Giải bài toán trong ảnh"} · Ảnh: ${file.name}` : question);
  input.value = ""; send.disabled = true; addLoading();
  try {
    const grade = $("#grade").value;
    const payload = { question: question || "Hãy đọc và giải bài toán trong ảnh.", curriculum: $("#curriculum").value, grade: grade ? Number(grade) : null };
    let body;
    if (file) {
      payload.image_base64 = await fileToBase64(file);
      try {
        body = await jsonRequest("/web/math/read-image", { method: "POST", body: JSON.stringify(payload) });
        $("#math-loading")?.remove(); addSolution(body);
      } catch (_) {
        const reply = await askGeneralChat(question || "Hãy mô tả và hỗ trợ nội dung trong ảnh này.", payload.image_base64);
        $("#math-loading")?.remove(); addChatAnswer(reply.text, "Ảnh không phải đề toán đã kiểm chứng nên được chuyển sang chatbot hình ảnh.", reply.mode);
      }
    } else {
      body = await jsonRequest("/web/math/route", { method: "POST", body: JSON.stringify(payload) });
      if (body.mode === "math") {
        $("#math-loading")?.remove(); addSolution(body.solution);
      } else if (body.mode === "math_unverified") {
        $("#math-loading")?.remove();
        addError("Chưa thể kiểm chứng bài toán", body.notice);
      } else {
        const reply = await askGeneralChat(question);
        $("#math-loading")?.remove();
        if (body.mode === "math_review") addMathReview(question, reply.text);
        else addChatAnswer(reply.text, body.notice || "Đã tự nhận biết đây là câu hỏi thông thường.", reply.mode || body.mode);
      }
    }
  } catch (error) {
    $("#math-loading")?.remove(); addError(error.message || "Không kết nối được máy chủ.", "Vui lòng thử lại sau.");
  } finally {
    send.disabled = false; imageInput.value = ""; $(".upload").classList.remove("has-file");
    $("#file-status").textContent = "Không dùng câu hỏi của bạn để tự huấn luyện. Ảnh chỉ xử lý trong bộ nhớ.";
    input.focus();
  }
});

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  $(".upload").classList.toggle("has-file", Boolean(file));
  $("#file-status").textContent = file
    ? `Đã chọn ${file.name}. Ảnh sẽ được OCR/VLM rồi kiểm chứng bằng bộ giải.`
    : "Không dùng câu hỏi của bạn để tự huấn luyện. Ảnh chỉ xử lý trong bộ nhớ.";
});

const roomConfig = {
  all: {
    breadcrumb: "Tổng quan",
    label: "Phòng học đa môn",
    hint: "AI tự nhận diện đề Toán, Văn hoặc Anh",
    title: "Hôm nay bạn muốn luyện gì?",
    copy: "Chọn một dạng bài hoặc nhập đề riêng ở khung bên cạnh.",
    placeholder: "Dán đề Toán, Văn, Tiếng Anh hoặc hỏi bất kỳ chủ đề nào…",
  },
  math: {
    breadcrumb: "Phòng thi Toán",
    label: "Phòng thi Toán • Lớp 1–12",
    hint: "Đáp án xác định có kiểm chứng và hình SVG",
    title: "Chọn dạng Toán cần luyện",
    copy: "Số học, đại số, đồ thị, hình học và giải tích THPT.",
    placeholder: "Nhập phép tính, phương trình, yêu cầu vẽ đồ thị hoặc gửi ảnh đề Toán…",
  },
  literature: {
    breadcrumb: "Phòng thi Ngữ văn",
    label: "Phòng thi Ngữ văn • Lớp 1–12",
    hint: "Đọc hiểu, nghị luận, phân tích và sáng tác",
    title: "Chọn dạng bài Ngữ văn",
    copy: "Gửi kèm ngữ liệu để phân tích và chấm bài sát đề nhất.",
    placeholder: "Dán ngữ liệu, đề đọc hiểu, đề văn hoặc yêu cầu sáng tác…",
  },
  english: {
    breadcrumb: "Phòng thi Tiếng Anh",
    label: "English Practice Room • Grade 1–12",
    hint: "Grammar, Reading, Writing và IELTS Task 1",
    title: "Choose your English practice",
    copy: "Luyện theo phong cách Task 1 Coach, có chữa lỗi và giải thích.",
    placeholder: "Paste an English test, passage, grammar question or Writing task…",
  },
};

const pageConfig = {
  chat: {
    room: "all",
    title: "Chatbot AI — hỏi mọi chủ đề",
    hero: "Một trang chat,<br><em>mọi câu hỏi của bạn.</em>",
    copy: "Hỏi tự nhiên về học tập, kiến thức, đời sống hoặc gửi ảnh. Khi nhận ra đề Toán, Văn hay Anh, hệ thống tự chuyển đúng gia sư mà bạn không cần chọn môn trước.",
    action: "Bắt đầu trò chuyện",
    jump: "tutor-workspace",
    verified: "AI tự nhận diện môn học",
  },
  math: {
    room: "math",
    title: "Phòng thi Toán lớp 1–12",
    hero: "Luyện Toán chắc từng bước,<br><em>nhìn thấy cả lời giải.</em>",
    copy: "Bộ đề từ số học, đại số, hình học đến giải tích. Bài trong miền xác định được tính bằng code, thế ngược và dựng hình SVG chuyển động theo từng bước.",
    action: "Chọn đề Toán",
    jump: "exam-library",
    verified: "✓ Toán có đối chiếu dữ kiện",
  },
  literature: {
    room: "literature",
    title: "Phòng thi Ngữ văn lớp 1–12",
    hero: "Đọc sâu một văn bản,<br><em>mở ra nhiều cách hiểu.</em>",
    copy: "Đọc hiểu, nghị luận xã hội, nghị luận văn học và sáng tác. Bài phân tích dài dùng nhiều lăng kính, phản đề, liên hệ đời sống và có bản đọc neural như podcast.",
    action: "Chọn đề Ngữ văn",
    jump: "exam-library",
    verified: "✓ Văn có nguồn và kiểm định",
  },
};

function examCard(exam, subject, index) {
  const article = el("article", `practice-card ${subject}-practice`);
  article.dataset.grade = String(exam.grade);
  const art = el("div", "practice-art");
  art.append(el("span", "practice-number", String(index + 1).padStart(2, "0")));
  art.append(el("b", "practice-symbol", subject === "math" ? "∑" : "文"));
  art.append(el("i", "practice-grid"));
  const copy = el("div", "practice-copy");
  const meta = el("div", "practice-meta");
  meta.append(el("span", "grade-tag", `Lớp ${exam.grade}`), el("span", "", `${exam.duration} phút`));
  copy.append(meta, el("small", "practice-topic", exam.topic), el("h3", "", exam.title), el("p", "", exam.prompt));
  const actions = el("div", "practice-actions");
  if (subject === "math") {
    const start = el("button", "practice-start", "Tự làm"); start.type = "button";
    const explain = el("button", "practice-explain", "Giải thích →"); explain.type = "button";
    start.addEventListener("click", () => openMathExam(exam, false));
    explain.addEventListener("click", () => openMathExam(exam, true));
    actions.append(start, explain);
  } else {
    const outline = el("button", "practice-start", "Gợi ý dàn ý"); outline.type = "button";
    const write = el("button", "practice-explain", "Viết bài →"); write.type = "button";
    outline.addEventListener("click", () => requestLiteratureOutline(exam));
    write.addEventListener("click", () => selectLiteratureExam(exam));
    actions.append(outline, write);
  }
  copy.append(actions); article.append(art, copy);
  return article;
}

function examsForRoute() {
  const bank = window.STUDYSCOPE_EXAMS || { math: [], literature: [] };
  if (route === "math") return { subject: "math", exams: bank.math || [] };
  if (route === "literature") return { subject: "literature", exams: bank.literature || [] };
  return { subject: "math", exams: [...(bank.math || []), ...(bank.literature || [])] };
}

function renderExamBank(grade = "all") {
  const target = $("#exam-grid");
  if (!target) return;
  const bank = examsForRoute();
  const filtered = bank.exams.filter((exam) => grade === "all" || exam.grade === Number(grade));
  target.replaceChildren();
  filtered.forEach((exam, index) => {
    const subject = route === "literature" || (route === "chat" && index >= (window.STUDYSCOPE_EXAMS?.math?.length || 0)) ? "literature" : "math";
    target.append(examCard(exam, subject, index));
  });
  $("#exam-count").textContent = `${filtered.length} bài luyện${grade === "all" ? " • lớp 1–12" : ` • lớp ${grade}`}`;
}

function selectGrade(grade) {
  document.querySelectorAll(".grade-pill").forEach((pill) => pill.classList.toggle("active", pill.dataset.grade === String(grade)));
  if (grade !== "all") $("#grade").value = String(grade);
  renderExamBank(String(grade));
}

function openMathExam(exam, explain) {
  $("#grade").value = String(exam.grade);
  input.value = explain
    ? `${exam.prompt}\n\nHãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.`
    : exam.prompt;
  document.getElementById("tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  if (explain) window.setTimeout(() => form.requestSubmit(), 420);
  else window.setTimeout(() => input.focus(), 420);
}

function selectLiteratureExam(exam) {
  selectedLiteratureExam = exam;
  $("#submission-grade").textContent = `Lớp ${exam.grade}`;
  $("#submission-title").textContent = exam.title;
  $("#submission-prompt").textContent = exam.prompt;
  $("#grade").value = String(exam.grade);
  document.getElementById("literature-submission")?.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => literatureAnswer?.focus(), 420);
}

function requestLiteratureOutline(exam) {
  $("#grade").value = String(exam.grade);
  input.value = `Lập dàn ý chi tiết cho đề Ngữ văn lớp ${exam.grade} sau, nêu luận điểm, loại dẫn chứng cần dùng và lỗi dễ mắc; chưa viết bài hoàn chỉnh:\n${exam.prompt}`;
  document.getElementById("tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => form.requestSubmit(), 420);
}

function requestLongLiteratureAnswer() {
  const source = (input.value.trim() || lastLiteratureTopic || selectedLiteratureExam?.prompt || "").slice(0, 850);
  if (!source) {
    input.placeholder = "Nhập chủ đề hoặc đề Văn trước, rồi bấm lại nút 2.500+ chữ…";
    input.focus();
    return;
  }
  lastLiteratureTopic = source;
  input.value = (
    "Hãy viết ngay một bài văn hoàn chỉnh khoảng 2800 chữ, tối thiểu 2500 chữ. "
    + "Không trả lời bằng khung ý, không xin xác nhận và không tự rút ngắn. Phân tích sâu, đa chiều, có phản đề, "
    + "liên hệ đời sống phù hợp và dùng nguồn khi đề liên quan tác phẩm hoặc sự kiện có thật.\n\n"
    + `Đề/chủ đề: ${source}`
  );
  document.getElementById("tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => form.requestSubmit(), 260);
}

function selectSubject(subject, shouldScroll = true) {
  const selected = roomConfig[subject] ? subject : "all";
  const config = roomConfig[selected];
  document.body.dataset.room = selected;
  $("#breadcrumb").textContent = config.breadcrumb;
  $("#room-label").textContent = config.label;
  $("#room-hint").textContent = config.hint;
  $("#bank-title").textContent = config.title;
  $("#bank-copy").textContent = config.copy;
  input.placeholder = config.placeholder;
  document.querySelectorAll(".nav-item[data-subject]").forEach((item) => item.classList.toggle("active", item.dataset.subject === selected));
  document.querySelectorAll(".subject-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.subject === selected));
  document.querySelectorAll(".example").forEach((example) => {
    example.hidden = selected !== "all" && example.dataset.subject !== selected;
  });
  if (shouldScroll) {
    document.getElementById(selected === "all" ? "overview" : "tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

document.querySelectorAll("[data-subject]").forEach((button) => button.addEventListener("click", () => {
  const subject = button.dataset.subject || "all";
  selectSubject(subject, true);
  if (button.dataset.problem) {
    input.value = button.dataset.problem;
    setTimeout(() => input.focus(), 450);
  }
}));

document.querySelectorAll("[data-jump]").forEach((button) => button.addEventListener("click", () => {
  document.getElementById(button.dataset.jump)?.scrollIntoView({ behavior: "smooth", block: "start" });
  if (button.dataset.jump === "tutor-workspace") setTimeout(() => input.focus(), 450);
}));

document.querySelectorAll("[data-upload]").forEach((button) => button.addEventListener("click", () => {
  document.getElementById("tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => imageInput.click(), 450);
}));

document.querySelectorAll(".grade-pill").forEach((button) => button.addEventListener("click", () => selectGrade(button.dataset.grade || "all")));
$("#long-writing")?.addEventListener("click", requestLongLiteratureAnswer);

let timerRemaining = 45 * 60;
let timerHandle = null;
const timerDisplay = $("#timer");
const timerToggle = $("#timer-toggle");
function renderTimer() {
  const minutes = Math.floor(timerRemaining / 60).toString().padStart(2, "0");
  const seconds = (timerRemaining % 60).toString().padStart(2, "0");
  timerDisplay.textContent = `${minutes}:${seconds}`;
}
timerToggle.addEventListener("click", () => {
  if (timerHandle) {
    clearInterval(timerHandle); timerHandle = null; timerToggle.textContent = "Tiếp tục"; return;
  }
  if (timerRemaining <= 0) timerRemaining = 45 * 60;
  timerToggle.textContent = "Tạm dừng";
  timerHandle = setInterval(() => {
    timerRemaining -= 1; renderTimer();
    if (timerRemaining <= 0) { clearInterval(timerHandle); timerHandle = null; timerToggle.textContent = "Làm lại"; }
  }, 1000);
});

const progressKey = `studyscope-progress-${new Date().toISOString().slice(0, 10)}`;
function renderProgress() {
  const count = Number(localStorage.getItem(progressKey) || 0);
  const target = document.querySelector(".progress-ring strong");
  if (target) target.textContent = String(count);
}
function recordCompleted() {
  const count = Number(localStorage.getItem(progressKey) || 0) + 1;
  localStorage.setItem(progressKey, String(count));
  renderProgress();
}

const today = new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit" }).format(new Date());
const todayLabel = document.querySelector(".today-head b");
if (todayLabel) todayLabel.textContent = today;
const route = location.pathname.endsWith("/literature") ? "literature" : location.pathname.endsWith("/math") ? "math" : "chat";
const page = pageConfig[route];
document.body.classList.add(`route-${route}`);
document.title = `StudyScope — ${page.title}`;
$("#hero-title").innerHTML = page.hero;
$("#hero-copy").textContent = page.copy;
$("#hero-primary .action-copy").textContent = page.action;
$("#hero-primary").dataset.jump = page.jump;
$(".verified-status").textContent = page.verified;
document.querySelectorAll(".nav-item[data-route]").forEach((item) => item.classList.toggle("active", item.dataset.route === route));
selectSubject(page.room, false);
if (route === "math") {
  $("#exam-library-title").textContent = "Kho đề Toán theo từng lớp";
  $("#exam-library-copy").textContent = "Mỗi bài có nút Giải thích để gửi thẳng sang gia sư; lời giải xác định được kiểm chứng và trực quan hóa.";
} else if (route === "literature") {
  $("#exam-library-title").textContent = "Kho đề Ngữ văn theo từng lớp";
  $("#exam-library-copy").textContent = "Chọn đề, viết bài ngay trên trang rồi nộp để AI chấm, sửa và gợi ý phát triển theo rubric.";
}
renderExamBank();
renderProgress();
input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });

if (literatureAnswer) {
  literatureAnswer.addEventListener("input", () => {
    $("#answer-count").textContent = `${literatureAnswer.value.length.toLocaleString("vi-VN")} / 12.000 ký tự`;
  });
}

if (literatureForm) literatureForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const answer = literatureAnswer.value.trim();
  if (!selectedLiteratureExam) {
    $("#submission-title").textContent = "Bạn cần chọn một đề trước khi nộp";
    document.getElementById("exam-library")?.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  if (answer.length < 20) return;
  const button = $("#submit-literature");
  button.disabled = true; button.firstChild.textContent = "Đang gửi bài để chấm ";
  addUser(`Đã nộp bài “${selectedLiteratureExam.title}” • Lớp ${selectedLiteratureExam.grade} • ${answer.length.toLocaleString("vi-VN")} ký tự`);
  addLoading();
  document.getElementById("tutor-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const session = await ensureChatSession();
    await jsonRequest("/web/literature/submit", {
      method: "POST",
      body: JSON.stringify({ ...session, grade: selectedLiteratureExam.grade, prompt: selectedLiteratureExam.prompt, answer }),
    });
    const reply = await waitForChatReply();
    $("#math-loading")?.remove();
    addChatAnswer(reply.text, "Đã chấm theo rubric 10 điểm, kèm lỗi cụ thể, hướng phát triển và bản sửa tham khảo.", reply.mode || "writing");
  } catch (error) {
    $("#math-loading")?.remove();
    addError(error.message || "Chưa gửi được bài để chấm.", "Bài làm vẫn còn trong ô soạn; bạn có thể thử lại.");
  } finally {
    button.disabled = false; button.firstChild.textContent = "Nộp bài để AI chấm ";
  }
});
