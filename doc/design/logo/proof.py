#!/usr/bin/env python3
"""Render the identity at the sizes it is actually used at."""
import pathlib, subprocess
HERE = pathlib.Path(__file__).resolve().parent
IMG = (HERE / ".." / ".." / "source" / "img").resolve()
HTML = """<!doctype html><meta charset="utf-8"><style>
 body{{margin:0;font:13px/1.4 "Liberation Sans",Arial,sans-serif;color:#12263A;
      background:#fff;width:1160px}}
 .r{{display:flex;align-items:center;gap:24px;padding:14px 24px;border-top:1px solid #e6ebef}}
 .l{{width:132px;color:#5a6b7a;font-size:11px;text-transform:uppercase;letter-spacing:.7px;flex:none}}
 .b{{padding:10px 14px;border-radius:6px;display:flex;align-items:center;gap:18px}}
 .w{{background:#fff;border:1px solid #e6ebef}} .d{{background:#0f1720}}
 .s{{background:#2980b9}} .c{{background:#2980b9;padding:10px}}
 .c>div{{background:#e8f1f8;border-radius:8px;padding:7px 12px;display:flex;align-items:center}}
 .m img{{filter:grayscale(1) contrast(1.4)}} h1{{font-size:18px;margin:20px 24px 4px}}
 .n{{color:#8b98a5;font-size:11px;margin:2px 24px}}
</style><h1>SPEAR identity — concept C, finalized</h1>
<div class="r"><div class="l">small mark</div>
 <div class="b w"><img src="{i}/spear-mark-small.svg" height="16">
  <img src="{i}/spear-mark-small.svg" height="24"><img src="{i}/spear-mark-small.svg" height="32">
  <img src="{i}/spear-mark-small.svg" height="48"><img src="{i}/spear-mark-small.svg" height="64"></div>
 <div class="b d"><img src="{i}/spear-mark-small.svg" height="16">
  <img src="{i}/spear-mark-small.svg" height="32"><img src="{i}/spear-mark-small.svg" height="64"></div></div>
<div class="n">16 · 24 · 32 · 48 · 64 px</div>
<div class="r"><div class="l">detailed mark</div>
 <div class="b w"><img src="{i}/spear-mark.svg" height="24"><img src="{i}/spear-mark.svg" height="32">
  <img src="{i}/spear-mark.svg" height="64"><img src="{i}/spear-mark.svg" height="128"></div>
 <div class="b w"><img src="{i}/spear-icon.svg" height="32"><img src="{i}/spear-icon.svg" height="64"></div></div>
<div class="n">detailed mark at 24 · 32 · 64 · 128 px, then the square icon variant</div>
<div class="r"><div class="l">sidebar lockup</div>
 <div class="b s"><img src="{i}/spear-logo-horizontal.svg" height="42"></div>
 <div class="b c"><div><img src="{i}/spear-logo-horizontal.svg" height="42"></div></div></div>
<div class="n">bare theme blue, then on the light card the CSS adds</div>
<div class="r"><div class="l">horizontal</div>
 <div class="b w"><img src="{i}/spear-logo-horizontal.svg" height="60"></div></div>
<div class="r"><div class="l">landing</div>
 <div class="b w"><img src="{i}/spear-logo.svg" height="104"></div></div>
<div class="r m"><div class="l">monochrome</div>
 <div class="b w"><img src="{i}/spear-logo-horizontal.svg" height="46">
  <img src="{i}/spear-mark-small.svg" height="32"><img src="{i}/spear-mark-small.svg" height="16"></div>
 <div class="b d"><img src="{i}/spear-mark-small.svg" height="32"></div></div>
"""
p = HERE / ".proof.html"
p.write_text(HTML.format(i=IMG.as_uri()))
subprocess.run(["google-chrome","--headless","--disable-gpu","--no-sandbox",
  "--hide-scrollbars","--force-device-scale-factor=2","--window-size=1160,640",
  f"--screenshot={HERE/'proof.png'}", f"file://{p}"], check=True, capture_output=True)
p.unlink(); print("  wrote proof.png")
