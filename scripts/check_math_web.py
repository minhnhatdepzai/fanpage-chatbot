"""Kiểm tra mọi đề mẫu trên Chromium thật qua CDP, lưu báo cáo và ảnh.

Khởi động Chromium với --remote-debugging-port=9225, rồi:
uv run python scripts/check_math_web.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import httpx
import websockets


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=9225)
    args = parser.parse_args()
    async with httpx.AsyncClient() as client:
        target = (await client.put(f"http://127.0.0.1:{args.port}/json/new?{args.base_url}/web/math")).json()
    out = Path("artifacts/math-quality")
    await asyncio.to_thread(out.mkdir, parents=True, exist_ok=True)
    sequence = 0
    errors = []
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=20_000_000) as ws:

        async def command(method: str, params: dict | None = None) -> dict:
            nonlocal sequence
            sequence += 1
            await ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("method") == "Runtime.exceptionThrown":
                    errors.append(message["params"])
                if message.get("id") == sequence:
                    if "error" in message:
                        raise RuntimeError(message["error"])
                    return message.get("result", {})

        async def evaluate(expression: str) -> dict | list:
            result = await command(
                "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
            )
            if result.get("exceptionDetails"):
                raise RuntimeError(result["exceptionDetails"])
            return result["result"].get("value")

        await command("Runtime.enable")
        await command("Page.enable")
        await command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False},
        )
        for _ in range(40):
            ready = await evaluate("({ready:!!window.STUDYSCOPE_EXAMS && !!window.STUDYSCOPE_SCENES})")
            if ready["ready"]:
                break
            await asyncio.sleep(0.25)
        report = await evaluate("""(async()=>{
          const report=[];
          for(const exam of window.STUDYSCOPE_EXAMS.math){
            const response=await fetch('/web/math/route',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:exam.prompt+' Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.',grade:exam.grade})});
            const body=await response.json();
            if(body.mode!=='math')throw new Error(exam.title+': '+JSON.stringify(body));
            addSolution(body.solution);
            const message=document.querySelector('#conversation .message:last-child');
            const svg=message.querySelector('svg');
            if(!svg || svg.querySelectorAll('rect,circle,path,line').length<1)throw new Error('Không có hình: '+exam.title);
            if(svg.innerHTML.includes('NaN') || svg.innerHTML.includes('Infinity'))throw new Error('Tọa độ không hữu hạn: '+exam.title);
            if(svg.textContent.trim()==='Biểu thức=')throw new Error('Minh họa trống: '+exam.title);
            await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
            message.querySelectorAll('.step-item')[body.solution.steps.length-1].click();
            const visible=[...svg.querySelectorAll('rect,circle,path,line')].filter(n=>getComputedStyle(n).visibility==='visible' && Number(getComputedStyle(n).opacity)>.5);
            if(!visible.length)throw new Error('Hình không nhìn thấy sau chọn bước: '+exam.title);
            if(body.solution.visual.type==='rectangle'){
              const r=svg.querySelector('rect');
              if(Math.abs(Number(r.getAttribute('width'))/Number(r.getAttribute('height'))-body.solution.visual.length/body.solution.visual.width)>1e-8)throw new Error('Hình chữ nhật sai tỉ lệ');
            }
            if(body.solution.visual.type==='groups'){
              const expected=Math.min(30,body.solution.visual.groups+(body.solution.visual.additional_groups||0));
              if(svg.querySelectorAll('.object-group').length!==expected)throw new Error('Vẽ thiếu nhóm: '+exam.title);
            }
            report.push({title:exam.title,answer:body.solution.answer,visual:body.solution.visual.type,steps:body.solution.steps.length,objects:svg.querySelectorAll('rect,circle,path,line').length});
            message.remove();
          }
          const card=[...document.querySelectorAll('.practice-card')].find(card=>card.querySelector('h3').textContent==='Dãy ghế hội trường');
          card.querySelector('.practice-explain').click();
          for(let i=0;i<60;i++){
            await new Promise(resolve=>setTimeout(resolve,100));
            if(document.querySelector('.seat-row'))break;
          }
          if(!document.querySelector('.seat-row'))throw new Error('Nút Giải thích không dựng được lời giải');
          document.getElementById('tutor-workspace').scrollIntoView({block:'start'});
          return report;
        })()""")
        await asyncio.sleep(0.4)
        before = await evaluate(
            "({active:document.querySelectorAll('.step-item.is-active').length,visible:document.querySelectorAll('.step-item.is-revealed').length,hiddenRows:document.querySelectorAll('.seat-row.scene-hidden').length})"
        )
        await evaluate("(()=>{document.querySelectorAll('.step-item')[4].click();return {ok:true}})()")
        await asyncio.sleep(0.4)
        after = await evaluate(
            "({steps:document.querySelectorAll('.step-item').length,rows:document.querySelectorAll('.seat-row').length,seats:document.querySelectorAll('.seat-row rect').length,hiddenRows:document.querySelectorAll('.seat-row.scene-hidden').length,visibleRows:[...document.querySelectorAll('.seat-row')].filter(x=>getComputedStyle(x).visibility==='visible' && Number(getComputedStyle(x).opacity)>.5).length,placeholder:document.querySelector('.math-visual').textContent.includes('Biểu thức')})"
        )
        assert before["active"] == 1 and before["hiddenRows"] > 0, before
        assert after["seats"] == 960 and after["rows"] == 15 and after["hiddenRows"] == 0, after
        assert after["visibleRows"] == 15, after
        await evaluate("(()=>{document.querySelector('.replay-animation').click();return {ok:true}})()")
        await asyncio.sleep(0.4)
        replay = await evaluate(
            "({hiddenRows:document.querySelectorAll('.seat-row.scene-hidden').length,step:document.querySelector('.step-item.is-active .step-num').textContent})"
        )
        assert replay["hiddenRows"] == 13 and replay["step"] == "1", replay
        await evaluate("(()=>{document.querySelector('.pause-animation').click();return {ok:true}})()")
        paused = await evaluate(
            "({paused:document.querySelector('.solution-stage').classList.contains('is-paused')})"
        )
        assert paused["paused"], paused
        await evaluate("(()=>{document.querySelector('.pause-animation').click();return {ok:true}})()")
        await asyncio.sleep(0.7)
        resumed = await evaluate(
            "({step:document.querySelector('.step-item.is-active .step-num').textContent})"
        )
        assert resumed["step"] == "2", resumed
        await evaluate("(()=>{document.querySelectorAll('.step-item')[4].click();return {ok:true}})()")
        # Hai hình chữ nhật (lưng và mặt ghế) cho mỗi một trong 480 ghế.
        await evaluate(
            "(()=>{const stage=document.querySelector('.solution-stage');stage.scrollIntoView({block:'center'});return {ok:true}})()"
        )
        picture = await command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        (out / "math-seats-desktop.png").write_bytes(base64.b64decode(picture["data"]))
        await command(
            "Runtime.evaluate",
            {"expression": "document.querySelector('.expand-animation').click()", "userGesture": True},
        )
        await asyncio.sleep(0.3)
        fullscreen = await evaluate("({expanded:!!document.fullscreenElement})")
        assert fullscreen["expanded"], fullscreen
        picture = await command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        (out / "math-seats-expanded.png").write_bytes(base64.b64decode(picture["data"]))
        await evaluate("document.exitFullscreen().then(()=>({ok:true}))")
        await command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
        )
        await asyncio.sleep(0.3)
        mobile = await evaluate(
            "({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,svgWidth:document.querySelector('.math-visual').getBoundingClientRect().width})"
        )
        assert mobile["scrollWidth"] <= mobile["width"] + 1, mobile
        assert not errors, errors
        report = {
            "examples": report,
            "passed": len(report),
            "animation_before": before,
            "animation_after": after,
            "replay": replay,
            "pause": paused,
            "resume": resumed,
            "fullscreen": fullscreen,
            "mobile": mobile,
            "runtime_errors": errors,
            "base_url": args.base_url,
        }
        (out / "browser-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        await command("Page.close")


if __name__ == "__main__":
    asyncio.run(main())
