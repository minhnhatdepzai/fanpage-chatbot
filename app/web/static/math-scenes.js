"use strict";

// Hình vector xác định: dựng từ dữ kiện và kết quả của bộ giải, không đoán bằng ảnh AI.
window.STUDYSCOPE_SCENES = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const colors = ["#7963cc", "#ef8a6c", "#2698a2", "#e4b53d"];
  function node(tag, attrs = {}, value) {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (value !== undefined) n.textContent = value;
    return n;
  }
  function caption(root, x, y, value, attrs = {}) {
    root.append(node("text", { x, y, fill: "#51476b", "font-size": 17, "font-family": "system-ui", ...attrs }, value));
  }
  function chair(root, x, y, size, color, step = 0) {
    const g = node("g", { transform: `translate(${x} ${y})`, "data-step": step, class: "lesson-object" });
    g.append(node("rect", { x: 0, y: 0, width: size, height: size * .48, rx: size * .15, fill: color }));
    g.append(node("rect", { x: 0, y: size * .58, width: size, height: size * .35, rx: size * .1, fill: color }));
    root.append(g);
  }
  function create(data, steps) {
    const supported = ["work_timeline", "arithmetic_sequence", "groups", "right_triangle", "cuboid", "bar_chart", "number_journey", "number_comparison", "scale_map", "integral_area", "fraction_sum", "formula"];
    if (!supported.includes(data.type)) return null;
    const svg = node("svg", { viewBox: "0 0 860 570", class: "math-visual lesson-scene", role: "img", "aria-label": "Hình minh họa toán học theo từng bước" });
    svg.append(node("rect", { width: 860, height: 570, rx: 18, fill: "#fcfbff" }));
    let rowGroups = [];
    let rowLabel = null;
    let selectedRow = 0;
    if (data.type === "work_timeline") {
      const left=105,right=755,top=80,bottom=360;
      const mapX=t=>left+t/data.days*(right-left),mapY=v=>bottom-v/data.total*(bottom-top);
      caption(svg,36,32,"KHỐI LƯỢNG THEO THỜI GIAN",{"font-size":20,"font-weight":800});
      caption(svg,28,65,"m³");caption(svg,775,392,"Ngày");
      svg.append(node("line",{x1:left,y1:bottom,x2:right+10,y2:bottom,stroke:"#9186a6","stroke-width":2}));
      svg.append(node("line",{x1:left,y1:bottom,x2:left,y2:top-10,stroke:"#9186a6","stroke-width":2}));
      [0,data.first,data.total].forEach(v=>{
        const y=mapY(v);svg.append(node("line",{x1:left,y1:y,x2:right,y2:y,stroke:"#ece7f4"}));
        caption(svg,left-12,y+5,String(v),{"text-anchor":"end","font-size":14});
      });
      const middle=[mapX(data.first_days),mapY(data.first)];
      const first=node("g",{"data-step":2,class:"lesson-object work-phase-first"});
      first.append(node("path",{d:`M${left} ${bottom} L${middle}`,fill:"none",stroke:colors[0],"stroke-width":5}));
      first.append(node("circle",{cx:middle[0],cy:middle[1],r:7,fill:colors[0]}));
      caption(first,middle[0],400,String(data.first_days),{"text-anchor":"middle"});
      caption(first,40,442,`Giai đoạn 1: ${data.first_days.toFixed(2).replace(/\.00$/,'')} ngày × ${data.initial_rate.toFixed(2).replace(/\.00$/,'')} m³/ngày = ${data.first} m³`,{"font-size":17,fill:colors[0]});svg.append(first);
      const second=node("g",{"data-step":3,class:"lesson-object work-phase-second"});
      second.append(node("path",{d:`M${middle} L${right} ${top}`,fill:"none",stroke:colors[2],"stroke-width":5}));
      second.append(node("circle",{cx:right,cy:top,r:7,fill:colors[2]}));
      caption(second,right,400,String(data.days),{"text-anchor":"middle"});
      caption(second,40,475,`Giai đoạn 2: ${data.second_days.toFixed(2).replace(/\.00$/,'')} ngày × ${data.later_rate.toFixed(2).replace(/\.00$/,'')} m³/ngày = ${data.remaining} m³`,{"font-size":17,fill:colors[2]});
      const point=node("circle",{r:8,fill:colors[1],stroke:"#fff","stroke-width":2,class:"work-moving-point"});
      const firstLength=Math.hypot(middle[0]-left,middle[1]-bottom),secondLength=Math.hypot(right-middle[0],top-middle[1]);
      point.append(node("animateMotion",{path:`M${left} ${bottom} L${middle} L${right} ${top}`,dur:"8s",repeatCount:"indefinite",calcMode:"linear",keyPoints:`0;${firstLength/(firstLength+secondLength)};1`,keyTimes:`0;${data.first_days/data.days};1`}));
      second.append(point);svg.append(second);
      const check=node("g",{"data-step":5,class:"lesson-object"});
      caption(check,40,525,`${data.first_days.toFixed(2).replace(/\.00$/,'')} + ${data.second_days.toFixed(2).replace(/\.00$/,'')} = ${data.days} ngày · ${data.first} + ${data.remaining} = ${data.total} m³`,{"font-size":19,"font-weight":800});svg.append(check);
    } else if (data.type === "arithmetic_sequence") {
      caption(svg, 32, 38, "HỘI TRƯỜNG · MỖI HÌNH LÀ MỘT GHẾ", { "font-size": 16, "font-weight": 800 });
      const shownRows = Math.min(Math.max(data.count, data.nth), 24);
      const maxSeats = Math.max(data.first, data.first + (shownRows - 1) * data.difference, 1);
      const seatGap = Math.min(2, 625 / maxSeats * .12);
      const seatSize = Math.min(13, 625 / maxSeats - seatGap);
      const rowGap = Math.min(25, 425 / shownRows);
      for (let r = 1; r <= shownRows; r++) {
        const count = data.first + (r - 1) * data.difference;
        const group = node("g", { "data-row": r, "data-step": r <= 2 ? 0 : 1, class: "seat-row" });
        const y = 62 + (r - 1) * rowGap;
        caption(group, 18, y + 12, String(r).padStart(2, "0"), { "font-size": 12 });
        for (let c = 0; c < Math.min(count, 160); c++) chair(group, 72 + c * (seatSize + seatGap), y, seatSize, r === data.nth ? colors[1] : colors[0]);
        caption(group, 807, y + 12, `${count} ghế`, { "font-size": 12, "text-anchor": "end", "font-weight": 700 });
        group.style.setProperty("--row-delay", `${(r - 1) * 35}ms`);
        rowGroups.push(group); svg.append(group);
      }
      const pairing = node("g", { "data-step": 2, class: "lesson-object" });
      pairing.append(node("path", { d: "M820 71 Q852 270 820 410", fill: "none", stroke: colors[2], "stroke-width": 3, "stroke-dasharray": "6 5" }));
      caption(pairing, 45, 493, `${data.first} + ${data.first + (data.count - 1) * data.difference} = ${data.first * 2 + (data.count - 1) * data.difference} ghế / cặp hàng`, { fill: colors[2], "font-size": 18, "font-weight": 800 });
      svg.append(pairing);
      rowLabel = node("text", { x: 45, y: 535, fill: "#5d4c9c", "font-size": 20, "font-weight": 800 }); svg.append(rowLabel);
      if (shownRows < Math.max(data.count, data.nth) || maxSeats > 160) caption(svg, 450, 535, "Hiển thị một phần; nhãn giữ số lượng thực.", { "font-size": 12 });
    } else if (data.type === "groups") {
      const total = data.groups + (data.additional_groups || 0);
      const shown = Math.min(total, 30), per = Math.min(Math.floor(data.per_group), 20);
      const columns = Math.min(shown, 5), rows = Math.ceil(shown / columns);
      const width = 780 / columns, height = Math.min(140, 410 / rows);
      caption(svg, 40, 35, `${data.groups} nhóm ban đầu · ${data.per_group} phần tử / nhóm`, {"font-weight":800});
      for (let i = 0; i < shown; i++) {
        const extra = i >= data.groups;
        const g = node("g", {transform:`translate(${40+(i%columns)*width} ${60+Math.floor(i/columns)*height})`, "data-step":extra?1:0, class:"lesson-object object-group"});
        g.append(node("rect", {width:width-12,height:height-12,rx:12,fill:extra?"#e5f5ee":"#eee9fd",stroke:extra?colors[2]:"#d6cbed","stroke-width":2}));
        const dots = node("g", {"data-step":data.boys_per_group!==undefined?2:0,class:"lesson-object"});
        const dotCols = Math.min(per,5), dotSize = Math.min(16,(height-45)/Math.max(1,Math.ceil(per/5)));
        for(let p=0;p<per;p++)dots.append(node("circle",{cx:18+(p%dotCols)*dotSize*1.4,cy:22+Math.floor(p/dotCols)*dotSize*1.4,r:dotSize*.33,fill:data.boys_per_group!==undefined&&p>=data.boys_per_group?colors[1]:colors[0]}));
        g.append(dots);
        caption(g,10,height-22,extra?`Thêm nhóm ${i-data.groups+1}`:`Nhóm ${i+1}`,{"font-size":12}); svg.append(g);
      }
      const result = node("g", {"data-step":Math.min(2,steps.length-1),class:"lesson-object"});
      caption(result,40,515,data.boys_per_group!==undefined?`Mỗi nhóm: ${data.boys_per_group} nam + ${data.girls_per_group} nữ`:`${data.groups} × ${data.per_group} = ${data.groups*data.per_group} phần tử`,{"font-size":20,"font-weight":800});
      if(data.additional_groups)caption(result,40,545,`Thêm ${data.additional_groups} nhóm: ${data.additional_groups*data.per_group} phần tử`,{fill:colors[2],"font-size":18});
      svg.append(result);
      if(total>shown || data.per_group>per)caption(svg,40,475,"Hình hiển thị một phần; nhãn ghi đầy đủ số lượng thực.",{"font-size":12});
    } else if (data.type === "right_triangle") {
      const scale = Math.min(480 / data.ac, 350 / data.ab);
      const a = [120, 430], b = [120, 430 - data.ab * scale], c = [120 + data.ac * scale, 430];
      const t = data.ab ** 2 / (data.ab ** 2 + data.ac ** 2);
      const h = [b[0] + (c[0] - b[0]) * t, b[1] + (c[1] - b[1]) * t];
      svg.append(node("path", { d: `M${a} L${b} L${c} Z`, fill: "#ece6fd", stroke: colors[0], "stroke-width": 4 }));
      svg.append(node("path", { d: `M${a[0]} ${a[1]-22} h22 v22`, fill: "none", stroke: colors[0], "stroke-width": 2 }));
      [a, b, c].forEach((p, i) => caption(svg, p[0] - 23, p[1] + 20, "ABC"[i], { "font-weight": 800 }));
      caption(svg, 35, 260, `AB = ${data.ab}`); caption(svg, 280, 470, `AC = ${data.ac}`);
      const hyp = node("g", { "data-step": 1, class: "lesson-object" }); caption(hyp, 400, 220, `BC = ${data.bc}`, { fill: colors[1], "font-weight": 800 }); svg.append(hyp);
      const altitude = node("g", { "data-step": 3, class: "lesson-object" }); altitude.append(node("line", { x1: a[0], y1: a[1], x2: h[0], y2: h[1], stroke: colors[2], "stroke-width": 4, "stroke-dasharray": "7 4" }));
      caption(altitude, h[0] + 12, h[1], "H"); caption(altitude, 420, 365, `AH = ${data.ah}`, { fill: colors[2], "font-weight": 800 }); svg.append(altitude);
    } else if (data.type === "cuboid") {
      const scale = Math.min(600/(data.length+.6*data.width),360/(data.height+.35*data.width));
      const x=110,y=450,w=data.length*scale,h=data.height*scale,dx=data.width*scale*.6,dy=data.width*scale*.35;
      const frontTop=y-h,waterTop=y-h*data.fill;
      const tank = node("g");
      tank.append(node("path", { d: `M${x} ${frontTop} h${w} l${dx} ${-dy} h${-w} Z M${x+w} ${frontTop} l${dx} ${-dy} v${h} l${-dx} ${dy} Z M${x} ${frontTop} h${w} v${h} h${-w} Z`, fill: "#f3f0fb", stroke: colors[0], "stroke-width": 3 }));
      const water = node("g", { "data-step": 2, class: "lesson-object" });
      water.append(node("rect", { x:x+2,y:waterTop,width:Math.max(0,w-4),height:h*data.fill,fill:"#66c5df",opacity:.7 }));
      water.append(node("path", {d:`M${x+w} ${waterTop} l${dx} ${-dy} v${h*data.fill} l${-dx} ${dy} Z M${x} ${waterTop} h${w} l${dx} ${-dy} h${-w} Z`,fill:"#66c5df",opacity:.55}));
      caption(water,x+w/2,y-h*data.fill/2,`${data.litres} lít`,{"text-anchor":"middle","font-weight":800});
      tank.append(water); svg.append(tank);
      caption(svg,x+w/2,y+30,`${data.length} m`,{"text-anchor":"middle"}); caption(svg,x+w+dx/2,y-dy/2+25,`${data.width} m`); caption(svg,x+w+dx+18,y-dy-h/2,`${data.height} m`);
      caption(svg, 65, 45, `Mức nước ${data.fill * 100}%`, { "font-size": 23, "font-weight": 800 });
    } else if (data.type === "bar_chart") {
      const max = Math.max(...data.bars.map((b) => b.value), data.mean || 0, 1);
      const gap = 680 / data.bars.length;
      data.bars.forEach((bar, i) => {
        const height = bar.value / max * 290;
        const g = node("g", { "data-step": i, class: "lesson-object" });
        g.append(node("rect", { x: 100 + i * gap, y: 405 - height, width: gap * .65, height, rx: 10, fill: colors[i % colors.length] }));
        caption(g, 100 + i * gap + gap * .325, 430, bar.label, { "text-anchor": "middle" });
        caption(g, 100 + i * gap + gap * .325, 390 - height, String(bar.value), { "text-anchor": "middle", "font-weight": 800 }); svg.append(g);
      });
      if (data.mean !== undefined) {
        const y = 405 - data.mean / max * 290; const mean = node("g", { "data-step": 2, class: "lesson-object" });
        mean.append(node("line", { x1: 70, y1: y, x2: 800, y2: y, stroke: colors[2], "stroke-width": 3, "stroke-dasharray": "7 4" }));
        caption(mean, 100, 510, `Chia đều: ${data.mean} mỗi nhóm`, { fill: colors[2], "font-weight": 800 }); svg.append(mean);
      }
    } else if (data.type === "number_journey" || data.type === "number_comparison") {
      const values = data.positions || data.markers.map((m) => m.x);
      const min = Math.min(0, ...values) - 2, max = Math.max(0, ...values) + 2;
      const map = (x) => 65 + (x-min)/(max-min)*735;
      svg.append(node("line", { x1: 60, y1: 330, x2: 805, y2: 330, stroke: "#b7aecf", "stroke-width": 3 }));
      values.forEach((v, i) => {
        const g = node("g", { "data-step": i, class: "lesson-object" });
        g.append(node("circle", { cx: map(v), cy: 330, r: 9, fill: colors[i % colors.length] }));
        caption(g, map(v), 370 + (i % 2)*25, String(v), { "text-anchor": "middle", "font-weight": 800 });
        if (data.positions && i) {
          const start = map(values[i-1]), end = map(v); g.append(node("path", { d: `M${start} 315 Q${(start+end)/2} ${160-i*20} ${end} 315`, stroke: colors[i % colors.length], "stroke-width": 4, fill: "none" }));
          caption(g, (start+end)/2, 175-i*20, `${v-values[i-1]>=0?"+":""}${v-values[i-1]}`, { fill: colors[i%colors.length], "text-anchor": "middle" });
        } svg.append(g);
      });
    } else if (data.type === "scale_map") {
      svg.append(node("rect", { x: 60, y: 85, width: 740, height: 350, rx: 20, fill: "#e9f4ef" }));
      svg.append(node("path", { d: "M80 180 Q250 250 400 125 T780 175 M160 105 Q250 350 720 400", stroke: "#ccd9dd", "stroke-width": 28, fill: "none" }));
      svg.append(node("line", { x1: 180, y1: 270, x2: 660, y2: 270, stroke: colors[0], "stroke-width": 4, "stroke-dasharray": "7 5" }));
      [180, 660].forEach((x, i) => {svg.append(node("circle", {cx:x,cy:270,r:12,fill:colors[i]}));caption(svg,x,315,"AB"[i],{"text-anchor":"middle"});});
      caption(svg, 420, 245, `${data.map_cm} cm · tỉ lệ 1:${data.factor}`, { "text-anchor": "middle" });
      const g=node("g",{"data-step":2,class:"lesson-object"});caption(g,420,495,`Thực tế: ${data.km} km`,{"text-anchor":"middle","font-size":24,"font-weight":800});svg.append(g);
    } else if (data.type === "integral_area") {
      const points=data.points; const yMax=Math.max(1,...points.map(p=>p.y)), yMin=Math.min(0,...points.map(p=>p.y));
      const mapX=x=>100+(x-data.lower)/(data.upper-data.lower)*630; const mapY=y=>440-(y-yMin)/(yMax-yMin)*340;
      const zeroY=mapY(0);
      svg.append(node("line",{x1:90,y1:zeroY,x2:790,y2:zeroY,stroke:"#a59ab9","stroke-width":2}));
      svg.append(node("line",{x1:100,y1:460,x2:100,y2:65,stroke:"#a59ab9","stroke-width":2}));
      const path=points.map((p,i)=>`${i?"L":"M"}${mapX(p.x)} ${mapY(p.y)}`).join(" ");
      const area=node("g",{"data-step":2,class:"lesson-object"});area.append(node("path",{d:`${path} L730 ${zeroY} L100 ${zeroY} Z`,fill:"#c4b8ef",opacity:.6}));svg.append(area);
      svg.append(node("path",{d:path,stroke:colors[0],"stroke-width":4,fill:"none"}));
      caption(svg,100,490,String(data.lower));caption(svg,730,490,String(data.upper));caption(svg,120,40,`f(x) = ${data.expression}`);
      const total=node("g",{"data-step":3,class:"lesson-object"});caption(total,460,530,`Tích phân = ${data.result}`,{"font-size":24,"font-weight":800,"text-anchor":"middle"});svg.append(total);
    } else if (data.type === "fraction_sum") {
      const maxDen = Math.min(48, data.denominator);
      const width = 660 / maxDen;
      for(let unit=0;unit<Math.ceil(data.numerator/data.denominator);unit++) {
        for(let p=0;p<maxDen;p++)svg.append(node("rect",{x:100+p*width,y:100+unit*100,width:width-2,height:60,rx:3,fill:unit*maxDen+p<data.numerator?colors[0]:"#ebe6f6","data-step":1,class:"lesson-object"}));
      }
      caption(svg,120,490,`${data.numerator}/${data.denominator}`,{"font-size":30,"font-weight":800});
    } else {
      const count=Math.min(steps.length,5);
      steps.slice(0,count).forEach((step,i)=>{ const g=node("g",{"data-step":i,class:"lesson-object"});
        g.append(node("rect",{x:45,y:50+i*94,width:770,height:80,rx:14,fill:i%2?"#e9f4ef":"#ece6fb"}));
        caption(g,65,80+i*94,step.title,{"font-weight":800});
        caption(g,65,106+i*94,(step.detail.length>85?step.detail.slice(0,82)+"…":step.detail),{"font-size":14});svg.append(g);
      });
    }
    function update(index, row = selectedRow) {
      selectedRow = row;
      svg.querySelectorAll("[data-step]").forEach((n) => n.classList.toggle("scene-hidden", Number(n.dataset.step) > index));
      rowGroups.forEach((g) => {
        const r=Number(g.dataset.row); const active=row ? r===row : r===data.nth && index>=1;
        g.classList.toggle("highlight-row", active); g.classList.toggle("dim-row", Boolean(row) && !active);
      });
      if(rowLabel) rowLabel.textContent=row ? `Hàng ${row}: ${data.first+(row-1)*data.difference} ghế` : index>=3 ? `Tổng ${data.count} hàng = ${data.total} ghế` : `uₙ = ${data.first} + (n − 1) × ${data.difference}`;
    }
    svg.updateLessonStep=update; update(0);
    return svg;
  }
  return {create};
})();
