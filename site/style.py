"""The stylesheet for the index page.

The study reports carry their own copy of this design system in each study's
`page_kit.py`, because a study directory has to stand on its own -- it is
copied in from wherever the work was done and must keep working there. The
index is the one page this repository owns, so it keeps its own copy of the
tokens and adds the card furniture the reports have no use for.

Tokens, typography and the `.panel` / `.facts` / `.chip` rules are the same as
the reports', so the landing page and the pages it links to read as one series.
Everything below the CARDS banner is new here.
"""

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap");
:root{
  --ground:#F6F8F7; --panel:#FFFFFF; --ink:#10171A; --muted:#5C6B68;
  --rule:#DDE4E1; --accent:#1F6F5C; --accent-soft:#E3EFEA;
  --win:#1F6F5C; --loss:#9E3B3B; --null:#7A8783;
  --shadow:0 1px 2px rgba(16,23,26,.06),0 8px 24px -12px rgba(16,23,26,.18);
  --shadow-lift:0 2px 4px rgba(16,23,26,.08),0 16px 40px -16px rgba(16,23,26,.28);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1416; --panel:#151E21; --ink:#E6EDEA; --muted:#93A5A0;
    --rule:#25332F; --accent:#6FC2A8; --accent-soft:#172824;
    --win:#6FC2A8; --loss:#E08A8A; --null:#7E8F8A;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
    --shadow-lift:0 2px 4px rgba(0,0,0,.5),0 18px 46px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#0E1416; --panel:#151E21; --ink:#E6EDEA; --muted:#93A5A0;
  --rule:#25332F; --accent:#6FC2A8; --accent-soft:#172824;
  --win:#6FC2A8; --loss:#E08A8A; --null:#7E8F8A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
  --shadow-lift:0 2px 4px rgba(0,0,0,.5),0 18px 46px -16px rgba(0,0,0,.8);
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
h3{font-size:1.16rem;margin-top:0}
p,ul,ol{margin:0}
li{margin-bottom:.35rem}
a{color:var(--accent)}
.eyebrow{font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
.lede{font-size:1.2rem;color:var(--muted);text-wrap:pretty}
.rule{height:1px;background:var(--rule);border:0;margin:1.2rem 0 .4rem}
.num,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,Menlo,monospace;font-size:.86em;
  background:var(--accent-soft);padding:.08em .3em;border-radius:2px}
.bleed{width:min(105ch,94vw);margin-inline:calc(50% - min(52.5ch,47vw));}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  padding:1.3rem 1.5rem;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:.7rem}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1rem;
  padding:1.1rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.fact{display:flex;flex-direction:column;gap:.1rem}
.fact b{font-family:ui-monospace,Menlo,monospace;font-size:1.35rem;font-variant-numeric:tabular-nums;
  font-weight:600;letter-spacing:-.02em}
.fact span{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.footnote{font-size:.88rem;color:var(--muted)}
.chip{display:inline-block;font-size:.63rem;letter-spacing:.06em;
  text-transform:uppercase;padding:.1rem .35rem;border-radius:2px;font-weight:700;
  border:1px solid var(--rule);color:var(--muted)}

/* --------------------------------------------------- THE OVERVIEW COMPARISON */
.tablewrap{overflow-x:auto;margin:1rem 0 .4rem}
table.overview{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--panel)}
table.overview th,table.overview td{padding:.4rem .6rem;text-align:right;
  border-bottom:1px solid var(--rule);white-space:nowrap}
table.overview thead th{font-weight:600}
table.overview tr:first-child th{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);text-align:center;border-bottom:1px solid var(--rule)}
table.overview tr:nth-child(2) th{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600;border-bottom:1.5px solid var(--rule)}
table.overview th[scope="row"]{text-align:left;font-weight:500;white-space:nowrap}
table.overview td.from{text-align:left;font-size:.76rem;color:var(--muted);
  font-family:ui-monospace,Menlo,monospace}
table.overview td.num{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
table.overview td.zero{color:var(--muted);opacity:.55}
table.overview tbody tr:hover,table.overview tr:hover{background:var(--accent-soft)}
.overfig{margin:1.4rem 0 .4rem;display:flex;flex-direction:column;gap:.5rem}
.overfig img{display:block;width:100%;height:auto;border:1px solid var(--rule);
  border-radius:3px;background:#fff}
.overfig figcaption{font-size:.86rem;color:var(--muted);max-width:74ch;
  margin-inline:auto;text-align:center;text-wrap:pretty}

/* ------------------------------------------------------------------ CARDS */
.cards{display:grid;gap:1.5rem;grid-template-columns:repeat(auto-fit,minmax(26rem,1fr));
  margin-top:.6rem}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  box-shadow:var(--shadow);display:flex;flex-direction:column;gap:.65rem;
  padding:1.5rem 1.6rem 1.2rem;position:relative;
  transition:box-shadow .18s ease,transform .18s ease,border-color .18s ease}
.card:hover,.card:focus-within{box-shadow:var(--shadow-lift);transform:translateY(-2px);
  border-color:var(--accent)}
.card h3{font-size:1.35rem;letter-spacing:-.01em}
.card h3 a{color:inherit;text-decoration:none}
.card h3 a::after{content:"";position:absolute;inset:0}
.card .q{font-size:1rem;color:var(--muted);font-style:italic;text-wrap:pretty;margin-top:-.25rem}
.card p{font-size:.94rem;text-wrap:pretty}
.card .finding{font-size:.94rem;background:var(--accent-soft);border-left:2px solid var(--accent);
  border-radius:0 3px 3px 0;padding:.7rem .9rem;text-wrap:pretty}
.card .finding b{color:var(--win)}
.card .caveat{font-size:.84rem;color:var(--muted);text-wrap:pretty}

/* The paper the study puts to the test: cited in full, not as a bare link. */
.papers{border:1px solid var(--rule);border-radius:3px;background:var(--ground);
  padding:.85rem 1rem .9rem}
.papers .plabel{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);font-weight:700;margin-bottom:.5rem}
.paperlist{list-style:none;padding:0;margin:0;display:grid;gap:.75rem}
.paperlist .paper{margin:0}
.paperlist .paper + .paper{border-top:1px solid var(--rule);padding-top:.75rem}
.paper .ptitle{font-family:"Newsreader","Iowan Old Style",Charter,Georgia,serif;
  font-size:1.02rem;font-weight:600;line-height:1.3;text-wrap:pretty;color:var(--ink)}
.paper .pauthors{font-size:.82rem;color:var(--muted);margin-top:.15rem;text-wrap:pretty}
.paper .pwhere{font-size:.82rem;margin-top:.2rem}
.paper .pvenue{font-style:italic;color:var(--muted)}
.paper .pdoi{font-family:ui-monospace,Menlo,monospace;font-size:.76rem;
  position:relative;z-index:1}
.card .cardfacts{display:grid;grid-template-columns:repeat(auto-fit,minmax(6.5rem,1fr));
  gap:.6rem;padding:.85rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.card .cardfacts b{font-family:ui-monospace,Menlo,monospace;font-size:1.05rem;font-weight:600;
  font-variant-numeric:tabular-nums;display:block;letter-spacing:-.02em}
.card .cardfacts span{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.card .meta{display:flex;flex-wrap:wrap;gap:.4rem;align-items:baseline;font-size:.82rem;
  color:var(--muted);margin-top:auto;padding-top:.3rem}
.card .meta a{position:relative;z-index:1}
.card .sep{color:var(--rule)}
.card .validates{font-size:.84rem;color:var(--muted);text-wrap:pretty}
.card .validates a{position:relative;z-index:1}
.card .go{font-weight:600;color:var(--accent)}
@media (max-width:700px){
  .bleed{width:100%;margin-inline:0}
  .cards{grid-template-columns:1fr}
}
"""
