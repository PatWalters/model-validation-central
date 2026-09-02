"""Shared furniture for the report pages.

Both reports in this repository are the same kind of document -- a comparison of
methods over the same endpoints, scored the same way -- so they should look like
one another and like the blog they accompany. What differs between them is the
argument, not the typography, so the stylesheet and the pieces of page assembly
that carry no argument live here and the narrative stays in the builders.

  06_build_page.py            the seven-method foundation-model comparison
  13_build_trimole_page.py    the five-method Trimole-Hybrid comparison
"""

import base64
import io

from PIL import Image

import config as cfg

FIG_WIDTH = 1700  # px; figures are downscaled to keep the page under the size limit

# Plot colours, so the pages and the figures read as one system. These match
# PALETTE in 05_report.py -- a method keeps its colour across both reports.
METHOD_COLOR = {
    "lgbm": "#4C72B0",
    "chemprop_st": "#C44E52",
    "chemprop": "#DD8452",
    "chemeleon": "#55A868",
    "megacl": "#8172B3",
    "monroe": "#937860",
    "moljepa": "#DA8BC3",
    "trimole": "#CCB974",
}

SHORT = {
    "lgbm": "LightGBM",
    "chemprop_st": "ChemProp ST",
    "chemprop": "ChemProp MT",
    "chemeleon": "CheMeleon",
    "megacl": "MEGA-CL",
    "monroe": "Monroe",
    "moljepa": "Mol-JEPA",
    "trimole": "Trimole",
}

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap");
:root{
  --ground:#F6F8F7; --panel:#FFFFFF; --ink:#10171A; --muted:#5C6B68;
  --rule:#DDE4E1; --accent:#1F6F5C; --accent-soft:#E3EFEA;
  --win:#1F6F5C; --loss:#9E3B3B; --null:#7A8783;
  --shadow:0 1px 2px rgba(16,23,26,.06),0 8px 24px -12px rgba(16,23,26,.18);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1416; --panel:#151E21; --ink:#E6EDEA; --muted:#93A5A0;
    --rule:#25332F; --accent:#6FC2A8; --accent-soft:#172824;
    --win:#6FC2A8; --loss:#E08A8A; --null:#7E8F8A;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --ground:#0E1416; --panel:#151E21; --ink:#E6EDEA; --muted:#93A5A0;
  --rule:#25332F; --accent:#6FC2A8; --accent-soft:#172824;
  --win:#6FC2A8; --loss:#E08A8A; --null:#7E8F8A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:17px; line-height:1.65; margin:0;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:74ch;margin:0 auto;padding:clamp(2rem,5vw,4.5rem) 1.5rem 6rem;
  display:flex;flex-direction:column;gap:1.35rem}
h1,h2,h3{font-family:"Newsreader","Iowan Old Style",Charter,"Palatino Linotype",Georgia,serif;
  font-weight:600; text-wrap:balance; margin:0; line-height:1.2}
h1{font-size:clamp(2.1rem,5vw,3rem);letter-spacing:-.015em}
h2{font-size:1.6rem;margin-top:2.4rem}
h3{font-size:1.16rem;margin-top:1.2rem}
p,ul,ol{margin:0}
li{margin-bottom:.35rem}
a{color:var(--accent)}
.eyebrow{font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
.lede{font-size:1.2rem;color:var(--muted);text-wrap:pretty}
.rule{height:1px;background:var(--rule);border:0;margin:1.2rem 0 .4rem}
.num,.sd,td.num,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}

/* full-bleed bands inside the prose column */
.bleed{width:min(105ch,94vw);margin-inline:calc(50% - min(52.5ch,47vw));}
.figscroll{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}
figure{margin:1rem 0 .4rem;display:flex;flex-direction:column;gap:.5rem}
figure img{display:block;width:100%;min-width:640px;height:auto}
figcaption{font-size:.86rem;color:var(--muted);max-width:74ch;margin-inline:auto;text-align:center}

.tablewrap{overflow-x:auto;margin:.8rem 0 .3rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:.42rem .6rem;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  font-weight:600;border-bottom:1.5px solid var(--rule);vertical-align:bottom}
th[scope="row"]{text-align:left;font-weight:500;font-family:ui-monospace,Menlo,monospace;font-size:.82rem}
tbody tr:hover{background:var(--accent-soft)}
td .sd{color:var(--muted);font-size:.78rem;margin-left:.3rem}
td.best .num{font-weight:700;color:var(--accent)}
.swatch{display:inline-block;width:.55rem;height:.55rem;border-radius:2px;margin-right:.35rem;
  vertical-align:baseline}
.swatch.big{width:.75rem;height:.75rem}
.chip{display:inline-block;margin-left:.4rem;font-size:.63rem;letter-spacing:.06em;
  text-transform:uppercase;padding:.05rem .3rem;border-radius:2px;font-weight:700}
.chip.best,.chip.win{background:var(--accent-soft);color:var(--win)}
.chip.tied{background:transparent;color:var(--muted);border:1px solid var(--rule)}
.chip.loss{background:transparent;color:var(--loss);border:1px solid var(--loss)}
.chip.null{background:transparent;color:var(--null);border:1px solid var(--rule)}

.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  padding:1.3rem 1.5rem;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:.7rem}
.tallies{display:grid;gap:.5rem}
.tally{display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap;
  padding-bottom:.4rem;border-bottom:1px solid var(--rule)}
.tally:last-child{border-bottom:0;padding-bottom:0}
.tallyname{font-weight:600}
.tallynums{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:.85rem;
  color:var(--muted);font-variant-numeric:tabular-nums}
.tallynums b{color:var(--ink)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1rem;
  padding:1.1rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.fact{display:flex;flex-direction:column;gap:.1rem}
.fact b{font-family:ui-monospace,Menlo,monospace;font-size:1.35rem;font-variant-numeric:tabular-nums;
  font-weight:600;letter-spacing:-.02em}
.fact span{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.missing{color:var(--loss);font-style:italic}
.footnote{font-size:.88rem;color:var(--muted)}
.repo{font-size:.92rem;background:var(--accent-soft);border:1px solid var(--rule);
  border-radius:3px;padding:.85rem 1.1rem}
.howto{font-size:.94rem}
.howto h3{margin-top:0;font-size:1.02rem}
.howto ul{padding-left:1.1rem;margin:0}
.howto li{margin-bottom:.3rem}
.howto b.k-best{color:var(--win)}
.howto b.k-tied{color:var(--null)}
.howto b.k-loss{color:var(--loss)}
sup.ref{font-size:.62em;line-height:0;vertical-align:super;margin-left:.08em;
  font-variant-numeric:normal;font-feature-settings:normal}
sup.ref a{text-decoration:none;font-weight:600}
sup.ref a:hover,sup.ref a:focus-visible{text-decoration:underline}
ol.refs{list-style:none;counter-reset:ref;padding:0;font-size:.88rem;
  display:grid;gap:.6rem}
ol.refs li{counter-increment:ref;position:relative;padding-left:2rem;margin:0;
  color:var(--muted);text-wrap:pretty}
ol.refs li::before{content:counter(ref);position:absolute;left:0;top:.05em;
  width:1.35rem;text-align:right;font-family:ui-monospace,Menlo,monospace;
  font-size:.8rem;font-weight:600;color:var(--accent)}
ol.refs li b{color:var(--ink);font-weight:600}
ol.refs li:target::before{color:var(--ink)}
ol.refs .venue{font-style:italic}
code{font-family:ui-monospace,Menlo,monospace;font-size:.86em;
  background:var(--accent-soft);padding:.08em .3em;border-radius:2px}
@media (max-width:700px){.bleed{width:100%;margin-inline:0}}
"""


def embed_figure(path) -> str:
    """A figure as a data URI, downscaled to FIG_WIDTH."""
    if not path.exists():
        return ""
    img = Image.open(path)
    if img.width > FIG_WIDTH:
        img = img.resize((FIG_WIDTH, round(img.height * FIG_WIDTH / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure_block(uri: str, caption: str, name: str) -> str:
    if not uri:
        return f'<p class="missing">figure {name} not found</p>'
    return (
        '<figure class="bleed">'
        f'<div class="figscroll"><img src="{uri}" alt="{caption}"></div>'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def reference_numbers(references) -> dict:
    """Reference key to its position in the list, which is its printed number."""
    return {key: n for n, (key, *_) in enumerate(references, start=1)}


def marker(numbers: dict, *keys: str) -> str:
    """A superscript marker linking into the reference list."""
    links = ", ".join(f'<a href="#ref-{key}">{numbers[key]}</a>' for key in keys)
    return f'<sup class="ref">{links}</sup>'


def render_references(references) -> str:
    """The numbered list itself, from (key, authors, title, detail, url, label)."""
    items = [
        f'<li id="ref-{key}">{authors}<b>{title}</b> {detail} '
        f'<a href="{url}">{label}</a></li>'
        for key, authors, title, detail, url, label in references
    ]
    return '<ol class="refs">' + "".join(items) + "</ol>"
