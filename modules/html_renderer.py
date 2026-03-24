"""
html_renderer.py — Professional AI HTML Presentation Renderer
===========================================================
Combines RAG-powered content with high-end interactive UI/UX.
Features: Particles, Particles interaction, Charts, TTS (stub), Q&A (stub), Professional Themes.
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path

log = logging.getLogger("renderer")

# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "obsidian": { "bg": "#09090f", "bg2": "#111118", "surf": "rgba(255,255,255,.05)",
                     "brd": "rgba(255,255,255,.08)", "ink": "#f0eeff", "muted": "rgba(240,238,255,.45)",
                     "a": "#7c5cfc", "a2": "#b06bff", "a3": "#4cc9f0",
                     "orb1": "#7c5cfc", "orb2": "#b06bff", "orb3": "#4cc9f0", "dark": True },
    "aurora":   { "bg": "#050e1a", "bg2": "#071524", "surf": "rgba(0,255,200,.04)",
                     "brd": "rgba(0,255,200,.10)", "ink": "#e8fff9", "muted": "rgba(232,255,249,.45)",
                     "a": "#00e5a0", "a2": "#00cfff", "a3": "#7b2fff",
                     "orb1": "#00e5a0", "orb2": "#00cfff", "orb3": "#7b2fff", "dark": True },
    "inferno":  { "bg": "#0d0600", "bg2": "#150900", "surf": "rgba(255,120,0,.05)",
                     "brd": "rgba(255,120,0,.10)", "ink": "#fff5ee", "muted": "rgba(255,245,238,.45)",
                     "a": "#ff6b00", "a2": "#ff3d6e", "a3": "#ffd600",
                     "orb1": "#ff6b00", "orb2": "#ff3d6e", "orb3": "#ffd600", "dark": True },
    "ocean":    { "bg": "#020c18", "bg2": "#041424", "surf": "rgba(56,189,248,.05)",
                     "brd": "rgba(56,189,248,.10)", "ink": "#e8f4ff", "muted": "rgba(232,244,255,.45)",
                     "a": "#38bdf8", "a2": "#818cf8", "a3": "#34d399",
                     "orb1": "#38bdf8", "orb2": "#818cf8", "orb3": "#34d399", "dark": True },
    "forest":   { "bg": "#030d05", "bg2": "#051408", "surf": "rgba(52,211,153,.05)",
                     "brd": "rgba(52,211,153,.10)", "ink": "#eeffee", "muted": "rgba(238,255,238,.45)",
                     "a": "#34d399", "a2": "#a3e635", "a3": "#fbbf24",
                     "orb1": "#34d399", "orb2": "#a3e635", "orb3": "#fbbf24", "dark": True },
    "neon":     { "bg": "#020207", "bg2": "#040410", "surf": "rgba(255,0,220,.05)",
                     "brd": "rgba(255,0,220,.10)", "ink": "#fff0ff", "muted": "rgba(255,240,255,.45)",
                     "a": "#ff00dc", "a2": "#00f5ff", "a3": "#ffff00",
                     "orb1": "#ff00dc", "orb2": "#00f5ff", "orb3": "#8b00ff", "dark": True },
    "galaxy":   { "bg": "#03000d", "bg2": "#070016", "surf": "rgba(167,139,250,.05)",
                     "brd": "rgba(167,139,250,.10)", "ink": "#f5f0ff", "muted": "rgba(245,240,255,.45)",
                     "a": "#a78bfa", "a2": "#f472b6", "a3": "#60a5fa",
                     "orb1": "#a78bfa", "orb2": "#f472b6", "orb3": "#60a5fa", "dark": True },
    "cream":    { "bg": "#f9f6f0", "bg2": "#f2ede3", "surf": "rgba(0,0,0,.04)",
                     "brd": "rgba(0,0,0,.08)", "ink": "#1a1208", "muted": "rgba(26,18,8,.50)",
                     "a": "#c2410c", "a2": "#7c3aed", "a3": "#0d9488",
                     "orb1": "#c2410c", "orb2": "#7c3aed", "orb3": "#0d9488", "dark": False },
}

# Map our PPTX theme names → HTML theme names
_PPTX_TO_HTML_THEME = {
    "Dark Navy":     "obsidian",
    "Ocean Blue":    "ocean",
    "Forest Green":  "forest",
    "Inferno":       "inferno",
    "Aurora":        "aurora",
    "Neon":          "neon",
    "Galaxy":        "galaxy",
    "Cream":         "cream",
}

_KEYS = list(THEMES.keys())

def _auto_theme(pres_id: str) -> str:
    h = int(hashlib.md5(pres_id.encode()).hexdigest()[:8], 16)
    return _KEYS[h % len(_KEYS)]

def _esc(s: str) -> str:
    return (str(s or "")).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"','&quot;').replace("'","&#39;").replace("\n"," ")

def _esc_mermaid(s: str) -> str:
    """
    Escape for Mermaid diagrams - minimal escaping to preserve syntax.
    Keeps > for arrows (-->) and prevents escaping & to avoid breaking mermaid.
    """
    if not s:
        return ""
    # Avoid escaping & and > to preserve --> and other mermaid syntax.
    return str(s).replace("<", "&lt;")

def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    if len(h) == 3: h = "".join([c*2 for c in h])
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _img_style(img: str | dict | None):
    if not img: return "", ""
    src = img.get("url") if isinstance(img, dict) else img
    if not src: return "", ""
    safe = src.replace("'", "%27")
    return (
        f' style="background-image:url(\'{safe}\');background-size:cover;background-position:center"',
        '<div class="img-overlay"></div>',
    )

# ── Chart JS builder ──────────────────────────────────────────────────────────
def _chart_js(cdata: dict, cid: str, t: dict) -> str:
    labels = cdata.get("labels", [])
    values = cdata.get("values", [])
    title  = cdata.get("title", "")
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    r,  g,  b  = _rgb(a)
    r2, g2, b2 = _rgb(a2)
    n = len(labels)
    is_dark = t["dark"]

    fills   = json.dumps([
        f"rgba({int(r+(r2-r)*i/max(n-1,1))},{int(g+(g2-g)*i/max(n-1,1))},{int(b+(b2-b)*i/max(n-1,1))},0.85)"
        for i in range(n)
    ])
    borders = json.dumps([a] * n)
    tick_c  = "rgba(240,238,255,.5)" if is_dark else "rgba(26,18,8,.5)"
    grid_c  = f"rgba({r},{g},{b},.07)"
    bg_c    = "rgba(6,5,15,.95)" if is_dark else "rgba(255,255,255,.95)"
    ink_c   = t["ink"]
    cb      = "(v)=>v>=1e9?(v/1e9).toFixed(1)+'B':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':String(v)"
    tip     = (f"tooltip:{{backgroundColor:'{bg_c}',titleColor:'{ink_c}',"
               f"bodyColor:'{tick_c}',borderColor:'{a}44',borderWidth:1,"
               f"padding:14,cornerRadius:12,displayColors:false,"
               f"titleFont:{{family:'Plus Jakarta Sans',weight:'700',size:13}},"
               f"bodyFont:{{family:'DM Mono',size:11}}}}")

    lj = json.dumps(labels)
    vj = json.dumps(values)
    tl = title.lower()
    ctype = "bar"
    if any(w in tl for w in ["trend","growth","year","history","evolution"]): ctype = "line"
    elif n <= 4 or any(w in tl for w in ["share","percent","breakdown","distribution"]): ctype = "doughnut"

    base = (f"responsive:true,maintainAspectRatio:false,"
            f"plugins:{{legend:{{display:{'true' if ctype=='doughnut' else 'false'}}},{tip}}}")

    if ctype == "bar":
        return (f"(function(){{var el=document.getElementById('{cid}');if(!el)return;"
                f"new Chart(el,{{type:'bar',data:{{labels:{lj},datasets:[{{data:{vj},"
                f"backgroundColor:{fills},borderColor:{borders},borderWidth:0,"
                f"borderRadius:10,borderSkipped:false}}]}},"
                f"options:{{{base},"
                f"animation:{{duration:1200,easing:'easeOutQuart',delay:(c)=>c.dataIndex*120}},"
                f"scales:{{x:{{ticks:{{color:'{tick_c}',font:{{family:'DM Mono',size:10}}}},"
                f"grid:{{display:false}}}},y:{{beginAtZero:true,"
                f"ticks:{{color:'{tick_c}',font:{{family:'DM Mono',size:10}},callback:{cb}}},"
                f"grid:{{color:'{grid_c}'}}}}}}}})}})();")

    if ctype == "line":
        return (f"(function(){{var el=document.getElementById('{cid}');if(!el)return;"
                f"var grd=el.getContext('2d').createLinearGradient(0,0,0,280);"
                f"grd.addColorStop(0,'rgba({r},{g},{b},.35)');grd.addColorStop(1,'rgba({r},{g},{b},0)');"
                f"new Chart(el,{{type:'line',data:{{labels:{lj},datasets:[{{data:{vj},"
                f"borderColor:'{a}',backgroundColor:grd,borderWidth:3,fill:true,tension:0.45,"
                f"pointBackgroundColor:'{a}',pointBorderColor:'{a2}',pointBorderWidth:2,"
                f"pointRadius:6,pointHoverRadius:9}}]}},"
                f"options:{{{base},"
                f"animation:{{duration:1600,easing:'easeInOutQuart'}},"
                f"scales:{{x:{{ticks:{{color:'{tick_c}',font:{{family:'DM Mono',size:10}}}},"
                f"grid:{{color:'{grid_c}'}}}},"
                f"y:{{ticks:{{color:'{tick_c}',font:{{family:'DM Mono',size:10}},callback:{cb}}},"
                f"grid:{{color:'{grid_c}'}}}}}}}})}})();")

    # doughnut
    r3, g3, b3 = _rgb(a3)
    multi = json.dumps([a, a2, a3, f"rgba({r},{g},{b},.6)", f"rgba({r2},{g2},{b2},.6)", f"rgba({r3},{g3},{b3},.6)"][:n])
    return (f"(function(){{var el=document.getElementById('{cid}');if(!el)return;"
            f"new Chart(el,{{type:'doughnut',data:{{labels:{lj},datasets:[{{data:{vj},"
            f"backgroundColor:{multi},borderColor:'rgba(0,0,0,.2)',borderWidth:2,"
            f"hoverOffset:14}}]}},"
            f"options:{{{base},cutout:'68%',"
            f"animation:{{animateRotate:true,animateScale:true,duration:1400,easing:'easeOutBack'}},"
            f"plugins:{{legend:{{display:true,position:'right',"
            f"labels:{{color:'{tick_c}',font:{{family:'DM Mono',size:10}},"
            f"padding:14,boxWidth:14,boxHeight:14,usePointStyle:true,pointStyle:'circle'}}}},"
            f"{tip}}}}}}})}})();")

# ── Slide HTML builders ───────────────────────────────────────────────────────
def _particles(idx: int, accent: str) -> str:
    return f'<canvas class="pc" id="pc{idx}" data-ac="{accent}"></canvas>'

def _slide_cover(s: dict, i: int, tot: int, t: dict, img: any) -> tuple:
    title = _esc(s.get("title", ""))
    notes = _esc(s.get("speaker_notes", ""))
    num   = f"{i+1:02d}/{tot:02d}"
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    bs, bo = _img_style(img)
    return (
        f'<section class="slide s-cover" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh" style="--c1:{a};--c2:{a2};--c3:{a3}"></div>'
        f'{bo}'
        f'{_particles(i, a)}'
        f'<div class="gridlines"></div>'
        f'<div class="intro-body">'
        f'  <div class="intro-tag"><span class="dot" style="background:{a}"></span>'
        f'  <span style="color:{a}">PRESENTATION</span>'
        f'  <span style="color:{a}44;margin-left:auto;font-size:9px">{num}</span></div>'
        f'  <h1 class="intro-h"><span class="shimmer">{title}</span></h1>'
        f'  <div class="glow-bar" style="background:linear-gradient(90deg,{a},{a2},{a3},transparent)"></div>'
        f'  <div class="meta-row">'
        f'    <span class="mk"><kbd>→</kbd> Next</span>'
        f'    <span class="mk"><kbd>P</kbd> Auto</span>'
        f'    <span class="mk"><kbd>Q</kbd> Q&amp;A</span>'
        f'    <span class="mk"><kbd>S</kbd> Speak</span>'
        f'  </div>'
        f'</div>'
        f'<div class="orbs">'
        f'  <div class="orb oa" style="background:radial-gradient(circle,{a}50,transparent 65%)"></div>'
        f'  <div class="orb ob" style="background:radial-gradient(circle,{a2}35,transparent 65%)"></div>'
        f'</div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), ""

def _slide_intro(s: dict, i: int, tot: int, t: dict, img: any) -> tuple:
    title = _esc(s.get("title", ""))
    bullets = s.get("bullets", [])
    para = " ".join([_esc(b) for b in bullets])
    notes = _esc(s.get("speaker_notes", ""))
    num   = f"{i+1:02d}/{tot:02d}"
    bs, bo = _img_style(img)
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    return (
        f'<section class="slide s-intro" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh subtle" style="--c1:{a};--c2:{a2};--c3:{a3}"></div>'
        f'{_particles(i, a)}'
        f'<div class="imgwrap"{bs}>{bo}</div>'
        f'<div class="inner" style="max-width:850px;text-align:center">'
        f'  <div class="eyebrow" style="margin-bottom:24px"><span class="etag" style="background:{a}18;border-color:{a}30;color:{a}">INTRODUCTION</span></div>'
        f'  <h2 class="stitle" style="font-size:clamp(32px,4vw,64px);margin-bottom:32px">{title}</h2>'
        f'  <div class="rule" style="background:linear-gradient(90deg,transparent,{a},{a2}88,transparent);margin:0 auto 36px"></div>'
        f'  <div class="intro-para" style="font-size:18px;line-height:1.8;color:var(--muted);font-weight:300;opacity:0;animation:fup .7s ease .4s both">{para}</div>'
        f'</div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), ""

def _slide_content(s: dict, i: int, tot: int, t: dict, img: any) -> tuple:
    title   = _esc(s.get("title", ""))
    bullets = s.get("bullets", [])
    notes   = _esc(s.get("speaker_notes", ""))
    num     = f"{i+1:02d}/{tot:02d}"
    a, a2, a3 = t["a"], t["a2"], t["a3"]

    bhtml = ""
    for j, b in enumerate(bullets):
        bull = b.get("text", "") or b.get("content", "") if isinstance(b, dict) else str(b)
        parts = bull.split(":", 1)
        kw    = parts[0].strip()
        rest  = parts[1].strip() if len(parts) > 1 else ""
        ac    = a2 if j % 2 else a
        ra, ga, ba = _rgb(ac)
        d  = 0.15 + j * 0.10
        bhtml += (
            f'<div class="brow" style="--d:{d:.2f}s">'
            f'<div class="bidx" style="background:rgba({ra},{ga},{ba},.15);border-color:rgba({ra},{ga},{ba},.35);color:{ac}">{j+1:02d}</div>'
            f'<div class="bbody"><span class="bkw" style="color:{ac}">{_esc(kw)}</span>'
            + (f'<span class="brest">{_esc(rest)}</span>' if rest else '')
            + f'</div></div>'
        )
    
    diag_html = ""
    if s.get("diagram"):
        diag_html = f'<div class="diagram-box"><pre class="mermaid">{_esc_mermaid(s["diagram"])}</pre></div>'

    img_html = ""
    if img:
        src = img.get("url") if isinstance(img, dict) else img
        if src:
            safe = src.replace("'", "%27").replace('"', '&quot;')
            img_html = f'<div class="inline-image-box"><img src="{safe}" class="inline-content-img" alt="Slide Image" /></div>'

    layout_class = "content-layout-split" if img_html else "content-layout-single"

    right_col = f'<div class="content-right">{img_html}</div>' if img_html else ""

    return (
        f'<section class="slide s-content" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh subtle" style="--c1:{a};--c2:{a2};--c3:{a3}"></div>'
        f'{_particles(i, a)}'
        f'<div class="rail" style="background:linear-gradient(180deg,{a},{a2}66,transparent)"></div>'
        f'<div class="inner">'
        f'  <div class="eyebrow"><span class="etag" style="background:{a}18;border-color:{a}30;color:{a}">SLIDE {i+1:02d}</span></div>'
        f'  <h2 class="stitle">{title}</h2>'
        f'  <div class="rule" style="background:linear-gradient(90deg,{a},{a2}88,transparent)"></div>'
        f'  <div class="{layout_class}">'
        f'    <div class="content-left"><div class="bullets">{bhtml}</div>{diag_html}</div>'
        f'    {right_col}'
        f'  </div>'
        f'</div>'
        f'<div class="orb oa sm" style="background:radial-gradient(circle,{a}20,transparent 65%)"></div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), ""

def _slide_stats(s: dict, i: int, tot: int, t: dict, img: any) -> tuple:
    title   = _esc(s.get("title", ""))
    bullets = s.get("bullets", [])
    cdata   = s.get("chart_data")
    notes   = _esc(s.get("speaker_notes", ""))
    num     = f"{i+1:02d}/{tot:02d}"
    bs, bo  = _img_style(img)
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    cid     = f"ch{i}"

    cards = ""
    for j, bull in enumerate(bullets[:4]):
        parts = bull.split(":", 1)
        kw    = parts[0].strip()
        rest  = parts[1].strip() if len(parts) > 1 else ""
        ac    = [a, a2, a3, a2][j % 4]
        ra, ga, ba = _rgb(ac)
        d = 0.15 + j * 0.09
        cards += (
            f'<div class="scard" style="--d:{d:.2f}s;border-color:rgba({ra},{ga},{ba},.25)">'
            f'<div class="scglow" style="background:radial-gradient(circle at top left,rgba({ra},{ga},{ba},.15),transparent 65%)"></div>'
            f'<div class="scnum" style="color:{ac}">{j+1:02d}</div>'
            f'<div class="sclbl" style="color:{ac}">{_esc(kw)}</div>'
            f'<div class="scval">{_esc(rest or kw)}</div>'
            f'</div>'
        )

    chart_html = js_chunk = ""
    if cdata:
        js_chunk = _chart_js(cdata, cid, t)
        chart_html = (
            f'<div class="cshell">'
            f'<div class="cglow" style="background:radial-gradient(circle,rgba({_rgb(a)[0]},{_rgb(a)[1]},{_rgb(a)[2]},.12),transparent 70%)"></div>'
            f'<canvas id="{cid}" class="ccanvas"></canvas>'
            f'</div>'
        )

    return (
        f'<section class="slide s-stats" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh subtle" style="--c1:{a};--c2:{a2};--c3:{a3}"></div>'
        f'{_particles(i, a)}'
        f'<div class="imgwrap"{bs}>{bo}</div>'
        f'<div class="inner">'
        f'  <div class="eyebrow"><span class="etag" style="background:{a}18;border-color:{a}30;color:{a}">DATA &amp; STATS</span></div>'
        f'  <h2 class="stitle">{title}</h2>'
        f'  <div class="rule" style="background:linear-gradient(90deg,{a},{a2}88,transparent)"></div>'
        f'  <div class="stats-layout"><div class="sgrid">{cards}</div>{chart_html}</div>'
        f'</div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), js_chunk

def _slide_comparison(s: dict, i: int, tot: int, t: dict, img: any) -> tuple:
    title  = _esc(s.get("title", ""))
    notes  = _esc(s.get("speaker_notes", ""))
    bullets = s.get("bullets", [])
    num    = f"{i+1:02d}/{tot:02d}"
    bs, bo = _img_style(img)
    a, a2, a3 = t["a"], t["a2"], t["a3"]

    mid = max(1, len(bullets) // 2)
    left_pts  = bullets[:mid]
    right_pts = bullets[mid:]

    def _col(pts, ac, side, delay):
        ra, ga, ba = _rgb(ac)
        items = "".join(f'<li class="cpi"><span class="cpbul" style="background:{ac}"></span>{_esc(p)}</li>' for p in pts)
        return (
            f'<div class="ccol {side}" style="--d:{delay}s">'
            f'<div class="chdr" style="background:rgba({ra},{ga},{ba},.12);border-color:rgba({ra},{ga},{ba},.30);color:{ac}">'
            f'{"Approach A" if side=="left" else "Approach B"}</div>'
            f'<ul class="clist">{items}</ul>'
            f'</div>'
        )

    return (
        f'<section class="slide s-comparison" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh subtle" style="--c1:{a};--c2:{a2};--c3:{a3}"></div>'
        f'{_particles(i, a)}'
        f'<div class="imgwrap"{bs}>{bo}</div>'
        f'<div class="inner">'
        f'  <div class="eyebrow"><span class="etag" style="background:{a}18;border-color:{a}30;color:{a}">COMPARISON</span></div>'
        f'  <h2 class="stitle">{title}</h2>'
        f'  <div class="rule" style="background:linear-gradient(90deg,{a},{a2}88,transparent)"></div>'
        f'  <div class="clayout">'
        f'    {_col(left_pts, a, "left", 0.20)}'
        f'    <div class="cdiv"><div class="vsbadge" style="background:linear-gradient(135deg,{a},{a2})">VS</div></div>'
        f'    {_col(right_pts, a2, "right", 0.40)}'
        f'  </div>'
        f'</div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), ""

def _slide_outro(s: dict, i: int, tot: int, t: dict, img: any, topic: str) -> tuple:
    title = _esc(s.get("title", "Conclusion"))
    notes = _esc(s.get("speaker_notes", ""))
    bullets = s.get("bullets", [])
    num   = f"{i+1:02d}/{tot:02d}"
    bs, bo = _img_style(img)
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    r, g, b = _rgb(a)
    
    # Build bullet HTML for conclusion points
    bhtml = ""
    for j, bu in enumerate(bullets[:6]):
        bull = bu.get("text", "") or bu.get("content", "") if isinstance(bu, dict) else str(bu)
        ac = a2 if j % 2 else a
        ra, ga, ba = _rgb(ac)
        d = 0.3 + j * 0.12
        bhtml += (
            f'<div class="outro-point" style="--d:{d:.2f}s;opacity:0;animation:fup .6s both var(--d)">'
            f'<span class="outro-bullet" style="background:{ac}"></span>'
            f'<span class="outro-text">{_esc(bull)}</span>'
            f'</div>'
        )
    
    bullets_section = f'<div class="outro-bullets">{bhtml}</div>' if bhtml else ''
    
    return (
        f'<section class="slide s-outro" data-idx="{i}" data-ac="{a}" style="--a:{a};--a2:{a2};--a3:{a3}">'
        f'<div class="mesh" style="--c1:{a3};--c2:{a};--c3:{a2}"></div>'
        f'{_particles(i, a2)}'
        f'<div class="gridlines"></div>'
        f'<div class="imgwrap"{bs}>{bo}</div>'
        f'<div class="outro-body">'
        f'  <div class="outro-ring"><svg viewBox="0 0 120 120" class="rsv"><circle cx="60" cy="60" r="54" fill="none" stroke="{a}" stroke-width="1" stroke-dasharray="4 8" opacity=".4"/><text x="60" y="67" text-anchor="middle" font-size="28" fill="{a}" font-weight="800">✦</text></svg></div>'
        f'  <h2 class="outro-h">{title}</h2>'
        f'  {bullets_section}'
        f'  <div class="outro-topic" style="color:{a2}">{_esc(topic)}</div>'
        f'  <p class="outro-cta">Press <kbd>Q</kbd> to take questions</p>'
        f'</div>'
        f'<div class="orbs"><div class="orb oa" style="background:radial-gradient(circle,{a}45,transparent 65%)"></div></div>'
        f'<div class="notes" data-n="{notes}">{s.get("speaker_notes","")}</div>'
        f'<div class="snum">{num}</div>'
        f'</section>'
    ), ""

# ── CSS ───────────────────────────────────────────────────────────────────────
def _css(t: dict) -> str:
    bg, bg2  = t["bg"], t["bg2"]
    surf, brd = t["surf"], t["brd"]
    ink, muted = t["ink"], t["muted"]
    a, a2, a3 = t["a"], t["a2"], t["a3"]
    r,g,b = _rgb(a)
    r2,g2,b2 = _rgb(a2)
    blend  = "screen" if t["dark"] else "multiply"
    return f"""
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--a:{a};--a2:{a2};--a3:{a3};--bg:{bg};--bg2:{bg2};--surf:{surf};--brd:{brd};--ink:{ink};--muted:{muted}}}
html,body{{width:100%;height:100%;overflow:hidden;background:{bg};color:{ink};font-family:'DM Sans',sans-serif;cursor:none}}
#cur{{position:fixed;width:12px;height:12px;border-radius:50%;background:{a};pointer-events:none;z-index:9999;transform:translate(-50%,-50%);transition:width .1s;mix-blend-mode:{blend}}}
#cur2{{position:fixed;width:36px;height:36px;border:1px solid {a};border-radius:50%;pointer-events:none;z-index:9998;transform:translate(-50%,-50%);opacity:.3}}
#deck{{position:fixed;inset:0}}
.slide{{position:absolute;inset:0;background:{bg};overflow:hidden;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transform:scale(2) translateZ(0);transition:none}}
.slide.active{{opacity:1;pointer-events:all;transform:scale(1) translateZ(0);transition:transform .8s cubic-bezier(.16,1,.3,1),opacity .4s}}
.slide.xl{{opacity:0;transform:scale(.5) translateX(-20%)}}
.slide.xr{{opacity:0;transform:scale(.5) translateX(20%)}}
.mesh{{position:absolute;inset:-50%;width:200%;height:200%;z-index:0;background:radial-gradient(ellipse at 20% 30%,var(--c1)15,transparent 60%),radial-gradient(ellipse at 80% 20%,var(--c2)10,transparent 60%),{bg};animation:mdrift 20s infinite alternate}}
@keyframes mdrift{{from{{transform:rotate(0deg)}}to{{transform:rotate(5deg) scale(1.05)}}}}
.gridlines{{position:absolute;inset:0;z-index:1;background-image:linear-gradient(rgba({r},{g},{b},.04) 1px,transparent 1px),linear-gradient(90deg,rgba({r},{g},{b},.04) 1px,transparent 1px);background-size:50px 50px}}
.pc{{position:absolute;inset:0;z-index:2;opacity:.5}}
.imgwrap{{position:absolute;inset:0;z-index:1;background-size:cover;background-position:center}}
.img-overlay{{position:absolute;inset:0;background:linear-gradient(rgba({r},{g},{b},.4),{bg} 90%)}}
.inner{{position:relative;z-index:5;max-width:1100px;width:100%;padding:60px}}
.stitle{{font-family:'Plus Jakarta Sans',sans-serif;font-size:clamp(30px,4vw,60px);font-weight:800;letter-spacing:-2px;margin-bottom:24px}}
.bullets{{display:flex;flex-direction:column;gap:15px}}
.brow{{display:flex;gap:20px;opacity:0;animation:fup .6s both var(--d)}}
@keyframes fup{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:none}}}}
.bidx{{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:10px;font-family:monospace;border:1px solid;flex-shrink:0}}
.bkw{{font-weight:700;display:block;font-size:1.1em;margin-bottom:4px}}
.brest{{color:{muted};font-weight:300}}
#hud{{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;justify-content:space-between;padding:15px 25px;pointer-events:none}}
.hbtns{{display:flex;gap:8px;pointer-events:all}}
.hb{{background:rgba({r},{g},{b},.1);border:1px solid rgba({r},{g},{b},.2);padding:8px 16px;border-radius:8px;color:{ink};cursor:none;font-size:11px;transition:all .2s;backdrop-filter:blur(10px)}}
.hb:hover{{background:rgba({r},{g},{b},.2)}}
.hb.on{{background:{a};color:white}}
#prog{{position:fixed;bottom:0;left:0;right:0;height:4px;z-index:100;background:rgba({r},{g},{b},.1)}}
#pf{{height:100%;background:linear-gradient(90deg,{a},{a2});transition:width .4s}}
#navp,#navn{{position:fixed;top:50%;transform:translateY(-50%);z-index:100;width:50px;height:50px;display:flex;align-items:center;justify-content:center;color:{muted};font-size:30px;cursor:none}}
#navp{{left:10px}}#navn{{right:10px}}
.notes{{display:none;position:fixed;bottom:0;left:0;right:0;background:rgba({r},{g},{b},.95);backdrop-filter:blur(20px);padding:30px 60px;font-family:monospace;z-index:150}}
.notes-on .notes{{display:block}}
.snum{{position:absolute;bottom:25px;right:30px;font-size:10px;color:{muted}}}
.diagram-box{{margin-top:25px;background:rgba(255,255,255,.03);border:1px solid rgba({r},{g},{b},.1);border-radius:12px;padding:20px;overflow:hidden;opacity:0;animation:fup .6s both .6s}}
.mermaid{{background:transparent !important;color:{ink} !important}}
.mermaid svg{{max-width:100%;height:auto;filter:drop-shadow(0 10px 20px rgba(0,0,0,.3))}}
.content-layout-split{{display:flex;gap:40px;align-items:center}}
.content-left{{flex:1}}
.content-right{{flex:1;display:flex;justify-content:center}}
.inline-image-box{{border-radius:16px;overflow:hidden;box-shadow:0 20px 40px rgba(0,0,0,0.4);border:1px solid rgba({r},{g},{b},0.3);max-height:500px;opacity:0;animation:fup .6s both .4s}}
.inline-content-img{{width:100%;height:auto;display:block;object-fit:cover;max-height:500px}}
.outro-bullets{{display:flex;flex-direction:column;gap:12px;margin:24px 0;max-width:700px}}
.outro-point{{display:flex;align-items:flex-start;gap:14px}}
.outro-bullet{{width:8px;height:8px;border-radius:50%;margin-top:8px;flex-shrink:0}}
.outro-text{{font-size:17px;line-height:1.6;color:{muted}}}
.s-outro{{text-align:center}}
.outro-body{{position:relative;z-index:5;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px;max-width:900px}}
.outro-ring{{width:100px;height:100px;margin-bottom:30px;animation:float 3s ease-in-out infinite}}
@keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-10px)}}}}
.rsv{{width:100%;height:100%}}
.outro-h{{font-family:'Plus Jakarta Sans',sans-serif;font-size:clamp(36px,5vw,72px);font-weight:800;letter-spacing:-2px;margin-bottom:16px;background:linear-gradient(135deg,{a},{a2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.outro-topic{{font-size:18px;margin-bottom:24px;opacity:0.7}}
.outro-cta{{font-size:13px;color:{muted};margin-top:20px}}
.outro-cta kbd{{background:{surf};border:1px solid {brd};padding:4px 10px;border-radius:6px;font-family:monospace}}
.orbs{{position:absolute;inset:0;pointer-events:none;overflow:hidden}}
.orb{{position:absolute;border-radius:50%;filter:blur(80px);animation:orb-float 8s ease-in-out infinite}}
.orb.oa{{width:400px;height:400px;bottom:-100px;right:-100px}}
@keyframes orb-float{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-20px,-20px)}}}}
"""

# ── JS ────────────────────────────────────────────────────────────────────────
_JS = r"""
(function(){
  function init(c){
    if(!c||c._i)return;c._i=true;
    const ac=c.dataset.ac||'#7c5cfc',ctx=c.getContext('2d'),sec=c.parentElement;
    let W=c.width=sec.offsetWidth,H=c.height=sec.offsetHeight;
    const rv=parseInt(ac.slice(1,3),16),gv=parseInt(ac.slice(3,5),16),bv=parseInt(ac.slice(5,7),16);
    const pts=Array.from({length:60},()=>({
      x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,r:Math.random()*2+1,a:Math.random()*.3+.1
    }));
    function draw(){
      ctx.clearRect(0,0,W,H);
      pts.forEach(p=>{
        p.x+=p.vx;p.y+=p.vy;
        if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
        ctx.save();ctx.globalAlpha=p.a;ctx.fillStyle=`rgba(${rv},${gv},${bv},1)`;
        ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();ctx.restore();
      });
      requestAnimationFrame(draw);
    }
    draw();
  }
  function initMermaid(){
    if(window.mermaid){
      mermaid.initialize({
        startOnLoad:false,
        theme:'dark',
        securityLevel:'loose',
        flowchart:{useMaxWidth:true,htmlLabels:true},
        suppressErrorRendering:true
      });
      // Render all mermaid diagrams on active slide
      setTimeout(function(){
        const active = document.querySelector('.slide.active');
        if(active){
          const diagrams = active.querySelectorAll('.mermaid');
          diagrams.forEach(function(el, idx){
            try {
              mermaid.render('mermaid-'+idx, el.textContent).then(function(result){
                el.innerHTML = result.svg;
              }).catch(function(err){
                console.warn('Mermaid render failed:', err);
                el.innerHTML = '<div style="color:#ff6b6b;padding:10px">Diagram unavailable</div>';
              });
            } catch(e) {
              console.warn('Mermaid error:', e);
            }
          });
        }
      }, 100);
    }
  }
  window._PI=init;
  window._IM=initMermaid;
})();

const SL=Array.from(document.querySelectorAll('.slide'));
let cur=0,notesOn=false;

function goTo(n,fwd=true){
  if(n<0||n>=TOT)return;
  SL[cur].classList.remove('active');
  SL[cur].classList.add(fwd?'xl':'xr');
  SL[n].classList.remove('xl','xr');
  SL[n].classList.add('active');
  const pc=SL[n].querySelector('.pc');if(pc)window._PI(pc);
  // Render mermaid diagrams on the new slide
  if(window.mermaid){
    const diagrams = SL[n].querySelectorAll('.mermaid:not([data-processed])');
    diagrams.forEach(function(el, idx){
      el.setAttribute('data-processed', 'true');
      const code = el.textContent;
      mermaid.render('mermaid-'+n+'-'+idx, code).then(function(result){
        el.innerHTML = result.svg;
      }).catch(function(err){
        console.warn('Mermaid render failed:', err);
        el.innerHTML = '<div style="color:#ff6b6b;padding:10px;font-size:12px">Diagram unavailable</div>';
      });
    });
  }
  cur=n;upd();
}
function fwd(){goTo(cur+1,true)}
function bwd(){goTo(cur-1,false)}
function upd(){
  document.getElementById('pf').style.width=((cur)/(TOT-1)*100)+'%';
  SL.forEach((s,i)=>s.classList.toggle('notes-on',notesOn&&i===cur));
}
function toggleNotes(){notesOn=!notesOn;upd();document.getElementById('btn-n').classList.toggle('on',notesOn);}
function toggleFS(){
  if(!document.fullscreenElement)document.documentElement.requestFullscreen();
  else document.exitFullscreen();
}
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key===' ')fwd();
  if(e.key==='ArrowLeft')bwd();
  if(e.key==='n')toggleNotes();
  if(e.key==='f')toggleFS();
});
document.addEventListener('mousemove',e=>{
  const c=document.getElementById('cur'),r=document.getElementById('cur2');
  c.style.left=e.clientX+'px';c.style.top=e.clientY+'px';
  r.style.left=e.clientX+'px';r.style.top=e.clientY+'px';
});
(function init(){
  SL[0].classList.add('active');
  const pc=SL[0].querySelector('.pc');if(pc)window._PI(pc);
  window._IM();
  upd();
})();
"""

def render(
    topic:      str,
    slides:     list,
    session_id: str,
    output_dir: str,
    images:     dict = None,
    theme_name: str = None,
) -> str:
    html_theme = _PPTX_TO_HTML_THEME.get(theme_name, theme_name)
    theme = THEMES.get(html_theme) or THEMES[_auto_theme(session_id)]
    total = len(slides)
    sections = []
    charts = []

    for i, s in enumerate(slides):
        stype = s.get("slide_type") or s.get("type") or "content"
        img = (images or {}).get(i) or (images or {}).get(s.get("image_id"))
        
        js_chart = ""
        if i == 0:     h, js_chart = _slide_cover(s, i, total, theme, img)
        elif i == total-1: h, js_chart = _slide_outro(s, i, total, theme, img, topic)
        elif stype == "comparison": h, js_chart = _slide_comparison(s, i, total, theme, img)
        elif stype == "stats" or s.get("chart_data"): h, js_chart = _slide_stats(s, i, total, theme, img)
        elif stype == "intro": h, js_chart = _slide_intro(s, i, total, theme, img)
        else:          h, js_chart = _slide_content(s, i, total, theme, img)
        
        sections.append(h)
        if js_chart: charts.append(js_chart)

    te = _esc(topic)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><title>{te} — Presentation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>{_css(theme)}</style></head>
<body><div id="cur"></div><div id="cur2"></div>
<div id="hud"><div id="title">{te}</div><div class="hbtns">
<button class="hb" id="btn-n" onclick="toggleNotes()">Notes</button>
<button class="hb" onclick="toggleFS()">FS</button></div></div>
<div id="prog"><div id="pf"></div></div>
<div id="navp" onclick="bwd()">‹</div><div id="navn" onclick="fwd()">›</div>
<div id="deck">{"".join(sections)}</div>
<script>const TOT={total}; {"".join(charts)} \n {_JS}</script></body></html>"""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pres_{session_id[:8]}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path