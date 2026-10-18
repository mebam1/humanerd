"""Generate self-contained static HTML research documentation."""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown


STYLE = r"""
:root{--bg:#f4f5f7;--ink:#17273a;--muted:#607184;--line:#dce3e9;--teal:#087f84;--nav:#112d3d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.8 'Segoe UI','Malgun Gothic',sans-serif}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}aside{position:fixed;inset:0 auto 0 0;width:276px;background:var(--nav);color:#eff8fa;padding:32px 24px;overflow:auto}
.brand{font-size:27px;letter-spacing:3px;font-weight:800;color:white}.brand small{display:block;font-size:11px;font-weight:400;letter-spacing:2px;color:#83bec6}
.nav-label{font-size:11px;letter-spacing:2px;color:#85a6b4;margin:32px 0 12px}nav a{display:block;color:#bbd1db;line-height:1.5;font-size:13px;padding:11px 13px;border-radius:7px;margin:5px 0}nav a.active,nav a:hover{color:#fff;background:#214856;text-decoration:none}
.side-note{font-size:11px;line-height:1.7;color:#85a6b4;border-top:1px solid #31515f;margin-top:28px;padding-top:20px}main{margin-left:276px;padding:38px 5vw 80px;max-width:1510px}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;color:var(--teal)}
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:28px;color:var(--muted);font-size:12px}.pill{border:1px solid #b9d9da;background:#e8f6f4;border-radius:99px;padding:3px 12px;color:#236d6b}
article{background:#fff;border:1px solid var(--line);border-radius:12px;padding:36px 44px;box-shadow:0 5px 18px #20344805;overflow-wrap:anywhere}h1{font-size:32px;letter-spacing:-1px;line-height:1.35;margin:0 0 28px}h2{font-size:21px;margin:38px 0 16px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:17px;margin-top:26px}p{margin:14px 0}strong{color:#123e4b}
table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{text-align:left;border-bottom:1px solid var(--line);padding:13px 15px;vertical-align:top}th{background:#edf4f6;white-space:nowrap;color:#234e5d}tr:nth-child(even) td{background:#fafcfd}
pre{background:#112b3a;color:#e1eff4;border-radius:8px;padding:20px;overflow:auto;font:12px/1.75 Consolas,monospace}code{font-family:Consolas,monospace;font-size:.88em;background:#edf2f5;color:#a43a54;padding:2px 5px;border-radius:4px}pre code{background:none;color:inherit;padding:0}
blockquote{margin:20px 0;border-left:4px solid #1c9d9e;background:#eef8f7;padding:8px 22px;color:#315866}li{margin:7px 0}img,video{max-width:100%;border-radius:8px}.toc{font-size:13px;border:1px solid var(--line);background:#f8fafb;border-radius:8px;padding:14px 24px;margin:20px 0}.toc ul{margin:3px 0}.toc li{margin:2px 0}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:26px 0}.card{display:block;border:1px solid var(--line);border-radius:9px;padding:22px;background:white;color:var(--ink)}.card:hover{border-color:var(--teal);text-decoration:none}.card b{display:block;font-size:16px}.card span{font-size:12px;color:var(--muted)}.hero{background:linear-gradient(120deg,#12384b,#086f77);color:#fff;border-radius:12px;padding:34px;margin-bottom:24px}.hero h1{margin:10px 0 14px}.hero p{color:#bfe1e5;max-width:780px}.hero .eyebrow{color:#8ee0da}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:24px}.stats div{border:1px solid #ffffff35;border-radius:8px;padding:12px 16px}.stats b{font-size:23px;display:block}.stats span{font-size:11px;color:#bfe1e5}footer{font-size:11px;color:var(--muted);margin-top:25px}
@media(max-width:950px){aside{position:relative;width:100%;padding:20px}nav{display:flex;flex-wrap:wrap}nav a{font-size:11px;padding:7px}.nav-label,.side-note{display:none}main{margin:0;padding:24px 18px}.brand small{display:inline;margin-left:12px}article{padding:25px}.cards{grid-template-columns:1fr}h1{font-size:27px}}
@media print{aside,.topbar{display:none}main{margin:0;padding:0}article{border:0;box-shadow:none}pre{white-space:pre-wrap}}
"""


def build_docs(root=None):
    root = Path(root or Path.cwd())
    out = root / "docs"
    out.mkdir(exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    (out / "assets" / "style.css").write_text(STYLE, encoding="utf-8")
    sources = sorted([*(root / "documentation").glob("*.md"), *out.glob("*.md")], key=lambda p: p.stem)
    pages = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), source.stem)
        pages.append((source.stem, title, text))

    def wrap(content, active):
        nav = '<a href="index.html" class="%s">연구 대시보드</a>' % ("active" if active == "index" else "")
        nav += "".join(f'<a href="{slug}.html" class="{"active" if slug == active else ""}">{html.escape(title)}</a>' for slug, title, _ in pages)
        title = next((title for slug, title, _ in pages if slug == active), "연구 대시보드")
        return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · CUBE LAB</title><link rel="stylesheet" href="assets/style.css"></head><body>
<aside><a class="brand" href="index.html">CUBE LAB<small>CONTACT DYNAMICS RESEARCH</small></a><div class="nav-label">RESEARCH NOTEBOOK</div><nav>{nav}</nav><div class="side-note">Point → Face → Body<br>Point → Patch → Link → Articulation<br><br>MuJoCo · NVIDIA Warp · PyTorch<br>코드와 실험 기록을 함께 읽는 연구 문서</div></aside>
<main><div class="topbar"><span>CUBE / RESEARCH IMPLEMENTATION</span><span class="pill">로컬 연구 환경</span></div>{content}<footer>CUBE CONTACT RESEARCH · 코드: src/cube · 설정: configs · 원본 제안: 연구 제안.md</footer></main></body></html>'''

    for slug, _, text in pages:
        text = re.sub(r"\]\(([^):]+)\.md\)", r"](\1.html)", text)
        md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
        body = md.convert(text)
        (out / f"{slug}.html").write_text(wrap(f"<article>{body}</article>", slug), encoding="utf-8")
    cards = "".join(f'<a class="card" href="{slug}.html"><span>RESEARCH NOTE {i:02d}</span><b>{html.escape(title)}</b><span>구현한 구성 요소 · 설계 이유 · 검증 방법 →</span></a>' for i, (slug, title, _) in enumerate(pages, 1))
    content = '''<section class="hero"><div class="eyebrow">HIERARCHICAL CONTACT DYNAMICS</div><h1>표면의 접촉에서<br>사족 로봇의 움직임까지.</h1><p>물리 포인트의 국소 표현과 학습 가능한 계층적 aggregation. Cube의 접촉 개입 실험부터 Ant의 일반화 관절 동역학과 모델 기반 제어까지 연결합니다.</p><div class="stats"><div><b>02</b><span>PHYSICS BACKENDS</span></div><div><b>04</b><span>BODY / ROBOT FAMILIES</span></div><div><b>06</b><span>EXPERIMENT STAGES · 0–5</span></div></div></section>'''
    content += '<article><div class="eyebrow">START HERE</div><h2 style="border:0;margin-top:8px">구현을 읽고, 결과를 확인하세요</h2><p>각 문서는 무엇을 구현했는지와 왜 필요한지를 함께 설명합니다. 실제 측정 결과는 실행 결과 문서에서 확인하고, 작은 검증 실험과 논문 수준의 성능 주장을 구분하세요.</p><pre><code>docker compose build research\ndocker compose run --rm research python -m cube --help\ndocker compose up -d docs</code></pre></article><div class="cards">' + cards + '</div>'
    (out / "index.html").write_text(wrap(content, "index"), encoding="utf-8")
    return {"pages": len(pages) + 1, "output": str(out), "url": "http://localhost:8000"}
