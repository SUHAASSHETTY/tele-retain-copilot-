"""Render the architecture diagram to portable image files.

Source:  docs/assets/architecture.mmd   (Mermaid flowchart)
Outputs: docs/assets/architecture.svg   (vector; plain SVG text, no HTML-in-SVG, so any viewer shows it)
         docs/assets/architecture.png   (raster, 2x scale)

Rendering uses a pinned Mermaid release in headless Chromium (Playwright). Only regenerating the
images needs network access (to fetch Mermaid); the included PNG/SVG are self-contained.

Run: python -m scripts.render_architecture
"""

from __future__ import annotations

import asyncio
import json
import sys

from src.config import ROOT_DIR

ASSETS = ROOT_DIR / "docs" / "assets"
SOURCE = ASSETS / "architecture.mmd"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;background:#ffffff;font-family:Helvetica,Arial,sans-serif}#out{display:inline-block;padding:24px}</style>
<script src="%s"></script></head><body><div id="out"></div>
<script>
mermaid.initialize({startOnLoad:false, theme:"default", securityLevel:"strict", htmlLabels:false,
  flowchart:{htmlLabels:false, curve:"basis", nodeSpacing:40, rankSpacing:55},
  themeVariables:{fontFamily:"Helvetica, Arial, sans-serif", fontSize:"15px"}});
window.renderDiagram = async (code) => {
  const {svg} = await mermaid.render("architecture", code);
  const out = document.getElementById("out");
  out.innerHTML = svg;
  const el = out.querySelector("svg");
  const vb = el.viewBox.baseVal;               // natural size: no percentage-width scaling in any viewer
  el.setAttribute("width", Math.ceil(vb.width));
  el.setAttribute("height", Math.ceil(vb.height));
  el.removeAttribute("style");
  el.setAttribute("style", "background-color:#ffffff");
  return el.outerHTML;
};
</script></body></html>"""


async def render() -> None:
    from playwright.async_api import async_playwright

    code = SOURCE.read_text()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=2, viewport={"width": 2400, "height": 2400})
        await page.set_content(PAGE % MERMAID, wait_until="networkidle")
        svg = await page.evaluate(f"window.renderDiagram({json.dumps(code)})")
        if "<foreignObject" in svg:
            raise RuntimeError("SVG contains HTML labels (foreignObject); expected plain SVG text")
        (ASSETS / "architecture.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg)
        box = await page.locator("#out").bounding_box()
        await page.set_viewport_size({"width": int(box["width"]) + 2, "height": int(box["height"]) + 2})
        await page.locator("#out").screenshot(path=str(ASSETS / "architecture.png"))
        await browser.close()
    print(f"docs/assets/architecture.svg ({(ASSETS / 'architecture.svg').stat().st_size // 1024} KB), "
          f"docs/assets/architecture.png ({(ASSETS / 'architecture.png').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(render())
    sys.exit(0)
