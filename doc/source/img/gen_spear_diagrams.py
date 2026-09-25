#!/usr/bin/env python3
# Generator for source/img/spear.drawio (multi-page) and the rendered diagrams.
#
#   python3 gen_spear_diagrams.py     # writes spear.drawio AND spear_<name>.svg
#   ./export_png.sh                   # optional PNG export, needs a working
#                                     # drawio CLI (see the note below)
#
# The .drawio file is the editable source of truth: open it in draw.io / the
# VS Code extension and every box is editable. This script regenerates it from
# a compact Python description so a structural change is made once instead of
# by hand in a dozen boxes.
#
# The SVGs are rendered directly from that same description. They exist because
# the drawio CLI cannot be relied upon: the drawio 30.4.1 snap fails on *every*
# input with "ReferenceError: next is not defined" from its own electron.js —
# including a five-element minimal file and the SO3 documentation's diagrams.
# Rendering here keeps the documentation build independent of that bug, and SVG
# stays crisp at any zoom. If you have a working drawio CLI and want PNGs too,
# export_png.sh is still there.
import base64
import html
import re
import textwrap


# ---- module sizes, read from the tree so they cannot go stale -------------
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "spear")


def loc(module):
    """`module.py — 1 532 l.` when the source is there, the bare name if not.

    The generator lives beside the documentation and the harness beside it;
    reading the real file means a module that doubles in size says so on the
    next regeneration instead of carrying a number somebody typed once.
    """
    path = os.path.join(_SRC, module + ".py")
    try:
        with open(path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
    except OSError:
        return module + ".py"
    return f"{module}.py — {n:,} l.".replace(",", " ")

# ---- palette -------------------------------------------------------------
CLI     = "fillColor=#dae8fc;strokeColor=#6c8ebf;"   # user-facing entry points
CORE    = "fillColor=#d5e8d4;strokeColor=#82b366;"   # SPEAR python core
MODEL   = "fillColor=#ffe6cc;strokeColor=#d79b00;"   # model serving
SANDBOX = "fillColor=#fff2cc;strokeColor=#d6b656;"   # confinement layers
DANGER  = "fillColor=#f8cecc;strokeColor=#b85450;"   # untrusted / host
STORE   = "fillColor=#e1d5e7;strokeColor=#9673a6;"   # persistent state
NEUTRAL = "fillColor=#f5f5f5;strokeColor=#999999;"
WHITE   = "fillColor=#ffffff;strokeColor=#666666;"
# ---- dark palette, for the page drawn in the style of the hand-made overview
INK     = "#0b1020"                                  # page background
D_CORE  = ("fillColor=#123a5c;strokeColor=#38bdf8;fontColor=#e6f6ff;")   # default on
D_OPT   = ("fillColor=#4a3209;strokeColor=#f59e0b;fontColor=#fff4dc;")   # optional
D_EXP   = ("fillColor=#3b1c53;strokeColor=#c084fc;fontColor=#f6e9ff;")   # experimental
D_SEC   = ("fillColor=#4c1220;strokeColor=#ef4444;fontColor=#ffe4e6;")   # security
D_BAND  = ("fillColor=#101a33;strokeColor=#334166;fontColor=#cfe3ff;")   # layer band
D_OUT   = ("fillColor=#132a22;strokeColor=#34d399;fontColor=#dcfce7;")   # outside
GHOST   = "fillColor=none;strokeColor=#444444;dashed=1;"

BOX  = "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
CONT = "rounded=1;whiteSpace=wrap;html=1;arcSize=4;verticalAlign=top;fontStyle=1;"
NOTE = "shape=note;whiteSpace=wrap;html=1;size=14;"
ARR  = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
        "strokeColor=#444444;fontSize=10;")
ARR_D = ARR + "dashed=1;"
ARR_R = ARR + "strokeColor=#b85450;"


PAGE_W, PAGE_H = 1100, 800


def _style_get(style, key, default=None):
    m = re.search(rf"(?:^|;){re.escape(key)}=([^;]*)", style)
    return m.group(1) if m else default


class Page:
    """One diagram page, emitted both as drawio XML and as standalone SVG."""

    def __init__(self, name, height=PAGE_H, background="#ffffff"):
        self.name = name
        self.background = background
        self.height = height
        self.cells = []
        self.shapes = {}   # id -> geometry/style record, for the SVG pass
        self.order = []    # drawing order
        self.links = []    # edges
        self.n = 1
        self.raster_only = False

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    @staticmethod
    def _v(label):
        # drawio renders \n only when encoded as a numeric char-ref
        return html.escape(label).replace("\n", "&#10;")

    def _record(self, i, x, y, w, h, label, style, fontsize, fontstyle):
        self.shapes[i] = {
            "x": x, "y": y, "w": w, "h": h, "label": label,
            "fill": _style_get(style, "fillColor", "none"),
            "stroke": _style_get(style, "strokeColor", "#444444"),
            "align": _style_get(style, "align", "center"),
            "valign": _style_get(style, "verticalAlign", "middle"),
            "rounded": _style_get(style, "rounded", "0") == "1",
            "dashed": _style_get(style, "dashed", "0") == "1",
            "text_only": style.startswith("text;"),
            "fontsize": fontsize,
            "bold": bool(fontstyle) or _style_get(style, "fontStyle") == "1",
            # A dark page needs light text, and drawio already carries that in
            # the style string; the SVG pass reads it from the same place
            # instead of taking a second parameter nobody would keep in sync.
            "fontcolor": _style_get(style, "fontColor", "#1a1a1a"),
        }
        self.order.append(i)

    def box(self, x, y, w, h, label, style=BOX, fontsize=12, fontstyle=0):
        i = self._id()
        s = f"{style}fontSize={fontsize};"
        if fontstyle:
            s += f"fontStyle={fontstyle};"
        self.cells.append(
            f'<mxCell id="{i}" value="{self._v(label)}" style="{s}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        self._record(i, x, y, w, h, label, s, fontsize, fontstyle)
        return i

    def label(self, x, y, w, h, text, fontsize=11, bold=False, align="center",
              color=None):
        # `color` exists for the dark page: free-standing text carries no fill
        # to inherit from, so on an ink background the default near-black is
        # invisible. Light pages pass nothing and keep the old value.
        st = ("text;html=1;whiteSpace=wrap;verticalAlign=middle;"
              f"align={align};fontSize={fontsize};{'fontStyle=1;' if bold else ''}"
              f"{f'fontColor={color};' if color else ''}")
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{self._v(text)}" style="{st}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        self._record(i, x, y, w, h, text, st, fontsize, 1 if bold else 0)
        return i

    def image(self, x, y, w, h, path):
        """Embed a raster file as a drawio image shape — a pasted picture.

        draw.io stores a pasted image as a base64 data URI inside the cell
        style, which is exactly what this writes, so the page opens with the
        picture in it and no dependency on a file next to the .drawio.

        The page carrying it is marked raster_only: the SVG pass would have to
        inline the same base64 a second time, and the documentation references
        the PNG directly anyway.
        """
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        i = self._id()
        style = ("shape=image;verticalLabelPosition=bottom;labelBackgroundColor="
                 "#ffffff;verticalAlign=top;aspect=fixed;imageAspect=0;"
                 f"image=data:image/png,{data};")
        self.cells.append(
            f'<mxCell id="{i}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        self.raster_only = True
        return i

    def note(self, x, y, w, h, text, fontsize=10, style=None):
        return self.box(x, y, w, h, text, NOTE + (style or
                        "fillColor=#ffffff;strokeColor=#999999;")
                        + "align=left;verticalAlign=top;", fontsize)

    def edge(self, src, dst, label="", style=ARR):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{self._v(label)}" style="{style}" '
            f'edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')
        self.links.append({
            "src": src, "dst": dst, "label": label,
            "dashed": "dashed=1" in style,
            "stroke": _style_get(style, "strokeColor", "#444444"),
        })
        return i

    # ---- drawio -----------------------------------------------------------

    def xml(self):
        body = "".join(self.cells)
        return (f'<diagram name="{html.escape(self.name)}" id="{self.name}">'
                f'<mxGraphModel background="{self.background}" '
                f'dx="1000" dy="700" grid="0" gridSize="10" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
                f'page="1" pageScale="1" pageWidth="{PAGE_W}" '
                f'pageHeight="{self.height}" '
                f'math="0" shadow="0"><root>'
                f'<mxCell id="0"/><mxCell id="1" parent="0"/>{body}'
                f'</root></mxGraphModel></diagram>')

    # ---- svg --------------------------------------------------------------

    @staticmethod
    def _wrap(text, width_px, fontsize):
        """Split on explicit newlines, then soft-wrap to the box width."""
        # 0.55 em per character is a good average for a sans-serif face.
        cols = max(8, int(width_px / (fontsize * 0.55)))
        lines = []
        for para in text.split("\n"):
            # Short lines are kept verbatim: textwrap would strip the leading
            # spaces that carry the hierarchy in the directory-layout page.
            if len(para) <= cols:
                lines.append(para)
            else:
                lines.extend(textwrap.wrap(para, cols) or [""])
        return lines

    def _anchor(self, rect, side):
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        return {
            "l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y), "b": (x + w / 2, y + h),
        }[side]

    def _route(self, a, b):
        """Orthogonal route between two rectangles, elbow in the middle."""
        ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        if bx >= ax + aw:                       # b clearly to the right
            p0, p1 = self._anchor(a, "r"), self._anchor(b, "l")
            mx = (p0[0] + p1[0]) / 2
            return [p0, (mx, p0[1]), (mx, p1[1]), p1]
        if bx + bw <= ax:                       # b clearly to the left
            p0, p1 = self._anchor(a, "l"), self._anchor(b, "r")
            mx = (p0[0] + p1[0]) / 2
            return [p0, (mx, p0[1]), (mx, p1[1]), p1]
        if by >= ay + ah:                       # b below
            p0, p1 = self._anchor(a, "b"), self._anchor(b, "t")
            my = (p0[1] + p1[1]) / 2
            return [p0, (p0[0], my), (p1[0], my), p1]
        p0, p1 = self._anchor(a, "t"), self._anchor(b, "b")
        my = (p0[1] + p1[1]) / 2
        return [p0, (p0[0], my), (p1[0], my), p1]

    def fitted_height(self, margin=24):
        """Trim the page to its content: no acre of white below the drawing."""
        if not self.shapes:
            return self.height
        return int(max(s["y"] + s["h"] for s in self.shapes.values()) + margin)

    def svg(self):
        height = self.fitted_height()
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {PAGE_W} {height}" width="{PAGE_W}" '
            f'height="{height}" font-family="Helvetica,Arial,sans-serif">',
            '<defs><marker id="ah" markerWidth="10" markerHeight="8" refX="9" '
            'refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" '
            'fill="#444444"/></marker></defs>',
            f'<rect width="{PAGE_W}" height="{height}" fill="{self.background}"/>',
        ]
        for i in self.order:
            s = self.shapes[i]
            if not s["text_only"]:
                rx = 8 if s["rounded"] else 0
                dash = ' stroke-dasharray="6 4"' if s["dashed"] else ""
                fill = s["fill"] if s["fill"] != "none" else "none"
                out.append(
                    f'<rect x="{s["x"]}" y="{s["y"]}" width="{s["w"]}" '
                    f'height="{s["h"]}" rx="{rx}" ry="{rx}" fill="{fill}" '
                    f'stroke="{s["stroke"]}" stroke-width="1.4"{dash}/>')
            fs = s["fontsize"]
            lh = fs * 1.32
            lines = self._wrap(s["label"], s["w"] - 12, fs)
            if s["valign"] == "top":
                y0 = s["y"] + 6 + fs
            else:
                y0 = s["y"] + s["h"] / 2 - (len(lines) - 1) * lh / 2 + fs * 0.35
            if s["align"] == "left":
                tx, anchor = s["x"] + 8, "start"
            elif s["align"] == "right":
                tx, anchor = s["x"] + s["w"] - 8, "end"
            else:
                tx, anchor = s["x"] + s["w"] / 2, "middle"
            weight = ' font-weight="bold"' if s["bold"] else ""
            for n, line in enumerate(lines):
                out.append(
                    f'<text x="{tx:.1f}" y="{y0 + n * lh:.1f}" '
                    f'font-size="{fs}" text-anchor="{anchor}" '
                    f'xml:space="preserve" '
                    f'fill="{s["fontcolor"]}"{weight}>{html.escape(line)}</text>')
        for link in self.links:
            a, b = self.shapes.get(link["src"]), self.shapes.get(link["dst"])
            if not a or not b:
                continue
            pts = self._route(a, b)
            d = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}"
                         for k, (px, py) in enumerate(pts))
            dash = ' stroke-dasharray="6 4"' if link["dashed"] else ""
            out.append(
                f'<path d="{d}" fill="none" stroke="{link["stroke"]}" '
                f'stroke-width="1.4"{dash} marker-end="url(#ah)"/>')
            if link["label"]:
                # Put the caption on the longest straight segment, with a small
                # white plate so it stays readable when it crosses a box edge.
                best, span = pts[0], -1.0
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    length = abs(x1 - x0) + abs(y1 - y0)
                    if length > span:
                        span, best = length, ((x0 + x1) / 2, (y0 + y1) / 2)
                width = 6.2 * len(link["label"]) + 8
                out.append(
                    f'<rect x="{best[0] - width / 2:.1f}" y="{best[1] - 13:.1f}" '
                    f'width="{width:.1f}" height="14" fill="#ffffff" '
                    f'fill-opacity="0.9" stroke="none"/>')
                out.append(
                    f'<text x="{best[0]:.1f}" y="{best[1] - 3:.1f}" font-size="10" '
                    f'text-anchor="middle" fill="#444444">'
                    f'{html.escape(link["label"])}</text>')
        out.append('</svg>')
        return "\n".join(out)


pages = []

# =========================================================================
# 1. Overview — the whole SPEAR stack
# =========================================================================
p = Page("overview")
p.label(0, 8, 1080, 30, "SPEAR — overall architecture", 16, True)

p.box(30, 55, 300, 200, "Entry points", CONT, 12, 1)
cli = p.box(55, 95, 250, 44, "spear-chat        (interactive REPL)", CLI, 10)
srv = p.box(55, 148, 250, 44, "spear-server   (llama-server unit)", CLI, 10)
mdl = p.box(55, 201, 250, 40, "spear-model   (model / adapter)", CLI, 10)

p.box(30, 285, 300, 250, "Persistent state", CONT, 12, 1)
chroma = p.box(55, 325, 250, 48, "chromadb/\nvector store (RAG corpus)", STORE, 10)
hist = p.box(55, 383, 250, 44, "history*.json\nconversation transcripts", STORE, 10)
audit = p.box(55, 437, 250, 44, "audit/tool-actions.jsonl\nmetadata-only audit trail", STORE, 10)

p.box(365, 55, 340, 480, "SPEAR core  (Python)", CONT, 12, 1)
rag = p.box(390, 95, 290, 66, "rag_chat.py\nREPL · prompt assembly · retrieval\ntool dispatch · UI", CORE, 10)
mb = p.box(390, 175, 290, 60, "model_backend.py\nprovider-neutral turns\nstop_reason contract", CORE, 10)
tr = p.box(390, 250, 290, 66, "tool_runtime.py\npolicy · workspace · sandbox\nresource control · audit", CORE, 10)
idx = p.box(390, 332, 290, 48, "index_corpus.py / index_dir.py\ncorpus ingestion", CORE, 10)
rules = p.box(390, 392, 290, 44, "rules.d/ · skills/ · system-prompt.md", NEUTRAL, 10)
proj = p.box(390, 448, 290, 44, "projects.json\nnamed workspaces", NEUTRAL, 10)

p.box(740, 55, 330, 230, "Model serving", CONT, 12, 1)
llama = p.box(765, 95, 280, 78, "llama-server  (llama.cpp-next)\n127.0.0.1:8080  ·  OpenAI-compatible\nspear-llm.service", MODEL, 10)
gguf = p.box(765, 185, 280, 40, "models/gguf/*.gguf", MODEL, 10)
anth = p.box(765, 235, 280, 38, "Anthropic API  (optional backend)", MODEL, 10)

p.box(740, 310, 330, 225, "Tool execution", CONT, 12, 1)
scope = p.box(765, 350, 280, 40, "systemd transient scope", SANDBOX, 10)
bwrap = p.box(765, 398, 280, 40, "bubblewrap sandbox", SANDBOX, 10)
cmd = p.box(765, 446, 280, 40, "COMMAND  (untrusted)", DANGER, 10)

p.edge(cli, rag)
p.edge(rag, mb)
p.edge(mb, llama, "HTTP")
p.edge(mb, anth, "", ARR_D)
p.edge(llama, gguf, "", ARR_D)
# rag_chat dispatches tool calls into tool_runtime; the two are stacked in the
# same column, so an explicit arrow would be drawn straight through
# model_backend. The adjacency carries the meaning.
p.edge(tr, scope, "tool call")
p.edge(scope, bwrap)
p.edge(bwrap, cmd)
p.edge(rag, chroma, "retrieval", ARR_D)
p.edge(tr, audit, "", ARR_D)
p.edge(idx, chroma, "", ARR_D)
p.edge(srv, llama)
pages.append(p)

# =========================================================================
# 2. Tool harness architecture
# =========================================================================
p = Page("harness")
p.label(0, 8, 1080, 30, "Tool execution harness  (tool_runtime.py)", 16, True)

p.box(30, 55, 500, 300, "Decision layer  —  what may run at all", CONT, 12, 1)
mode = p.box(55, 95, 210, 50, "ExecutionMode\nSAFE · ASK · AUTO", CORE, 10)
cap = p.box(285, 95, 220, 50, "CapabilityPolicy\nper-mode capability set", CORE, 10)
pol = p.box(55, 165, 450, 52, "CommandPolicy.assess()\nargv classification → CommandAssessment", CORE, 10)
auth = p.box(55, 232, 450, 52, "AuthorizationResult\ngranted capabilities · confirmation required", CORE, 10)
prof = p.box(55, 297, 450, 44, "ExecutionProfile\nworkspace_read/write · shell_complex · network", CORE, 10)

p.box(30, 375, 500, 175, "Boundary layer  —  where it runs", CONT, 12, 1)
ws = p.box(55, 415, 210, 50, "Workspace\ncanonical root, no escape", CORE, 10)
runner = p.box(285, 415, 220, 50, "CommandRunner\nowns the contracts", CORE, 10)
audit2 = p.box(55, 480, 450, 46, "AuditLogger  —  metadata only, secrets redacted", NEUTRAL, 10)

p.box(560, 55, 510, 495, "Execution layer  —  how it is confined", CONT, 12, 1)
sandbox = p.box(585, 95, 460, 50, "BubblewrapSandbox\nbuild_argv() · run() · _run_with_slirp()", SANDBOX, 10)
rl = p.box(585, 160, 220, 66, "ResourceLimits\nper-process rlimits\nvia prlimit, inside bwrap", SANDBOX, 10)
cl = p.box(825, 160, 220, 66, "CgroupLimits\nwhole-tree kernel limits\nvia systemd scope", SANDBOX, 10)
scoper = p.box(585, 240, 460, 56, "SystemdScopeRunner\nunit_name() · wrap() · availability_for() · terminate()", SANDBOX, 10)
avail = p.box(585, 310, 220, 60, "SandboxAvailability\nUNKNOWN/AVAILABLE/\nABSENT/REFUSED", NEUTRAL, 9)
cavail = p.box(825, 310, 220, 60, "CgroupAvailability\nSYSTEMD_RUN_ABSENT\nUSER_BUS_UNAVAILABLE …", NEUTRAL, 9)
net = p.box(585, 385, 460, 66, "slirp4netns attachment\npinned ns/net + NS_GET_USERNS owner\ninfo-fd · block-fd · sync-fd · ready-fd · exit-fd", SANDBOX, 10)
res = p.box(585, 465, 460, 60, "ToolResult\nstatus · stdout · stderr · exit_code · changed_paths", CORE, 10)

p.edge(mode, pol)
p.edge(cap, pol)
p.edge(pol, auth)
p.edge(auth, prof)
p.edge(prof, runner)
p.edge(ws, runner)
p.edge(runner, sandbox)
p.edge(sandbox, rl)
p.edge(sandbox, cl)
p.edge(cl, scoper)
# The remaining boxes of this column are stacked in execution order; drawing
# arrows through them would cross the intermediate boxes for no added meaning.
p.edge(runner, audit2, "", ARR_D)
pages.append(p)

# =========================================================================
# 3. Security model — modes and capabilities
# =========================================================================
p = Page("security")
p.label(0, 8, 1080, 30, "Security model  —  modes, capabilities, fail-closed", 16, True)

p.box(30, 55, 1040, 210, "DEFAULT_CAPABILITY_POLICY", CONT, 12, 1)
head = ["capability", "SAFE", "ASK", "AUTO"]
xs = [60, 420, 620, 820]
ws_ = [350, 190, 190, 190]
for x, w, h in zip(xs, ws_, head):
    p.box(x, 90, w, 32, h, NEUTRAL, 11, 1)
rows = [
    ("filesystem:read", "yes", "yes", "yes"),
    ("workspace:write", "no", "yes", "yes"),
    ("shell:complex", "no", "yes", "yes"),
    ("network", "no", "yes", "no"),
]
for r, row in enumerate(rows):
    y = 126 + r * 33
    p.box(xs[0], y, ws_[0], 30, row[0], WHITE, 10)
    for c in range(1, 4):
        style = CORE if row[c] == "yes" else DANGER
        p.box(xs[c], y, ws_[c], 30, row[c], style, 10)

p.label(60, 262, 1000, 22,
        "gpu · ssh · remote:write · container-runtime · secrets  —  declared but "
        "not implemented: build_argv() refuses them before execution", 10)

p.box(30, 300, 1040, 240, "Authorization pipeline", CONT, 12, 1)
s1 = p.box(60, 345, 180, 62, "model proposes\na tool call", DANGER, 10)
s2 = p.box(262, 345, 180, 62, "CommandPolicy\nclassify argv", CORE, 10)
s3 = p.box(464, 345, 180, 62, "capability\nintersection", CORE, 10)
s4 = p.box(666, 345, 180, 62, "confirmation\n(ASK mode)", CORE, 10)
s5 = p.box(868, 345, 172, 62, "sandboxed\nexecution", SANDBOX, 10)
p.edge(s1, s2)
p.edge(s2, s3)
p.edge(s3, s4)
p.edge(s4, s5)
deny = p.box(464, 445, 380, 70,
             "DENIED  —  any of: unknown classification, capability not granted, "
             "sandbox unavailable, resource control unavailable, user refusal",
             DANGER, 10)
p.edge(s3, deny, "", ARR_R)
p.edge(s4, deny, "", ARR_R)
p.note(60, 445, 380, 70,
       "Fail-closed is the rule everywhere:\nan unavailable mechanism is an "
       "error, never a downgrade to a\nless confined execution.", 10)
pages.append(p)

# =========================================================================
# 4. Sandbox — wrapper order and confinement
# =========================================================================
p = Page("sandbox")
p.label(0, 8, 1080, 30, "Sandbox  —  wrapper order and confinement", 16, True)

p.label(40, 48, 1000, 22,
        "The order is fixed. prlimit stays INSIDE bubblewrap; the cgroup scope "
        "stays OUTSIDE it.", 11)

y = 82
NEST = CONT + SANDBOX        # container styling + the sandbox palette
NEST_CMD = CONT + DANGER
w1 = p.box(40, y, 1000, 330, "systemd transient scope   spear-tool-<uuid>.scope",
           NEST, 12, 1)
p.label(60, y + 30, 500, 20, "MemoryMax · MemorySwapMax · TasksMax · CPUQuota",
        10, False, "left")
w2 = p.box(80, y + 58, 920, 250, "bwrap  (bubblewrap)", NEST, 12, 1)
p.label(100, y + 88, 700, 20,
        "--unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-net",
        10, False, "left")
p.label(100, y + 110, 700, 20,
        "--clearenv --die-with-parent --new-session", 10, False, "left")
w3 = p.box(120, y + 138, 840, 152, "prlimit  --nofile=4096:4096  --core=0:0",
           NEST, 11, 1)
w4 = p.box(160, y + 180, 760, 96, "COMMAND   (untrusted argv)", NEST_CMD, 12, 1)
p.label(180, y + 218, 720, 40,
        "PATH=/usr/bin:/bin   HOME=/home/sandbox   TMPDIR=/tmp   cwd=/workspace",
        10)

p.box(40, 440, 500, 240, "Filesystem view", CONT, 12, 1)
mounts = [
    ("/usr", "read-only bind of the host /usr"),
    ("/bin /lib /lib64", "symlinks into /usr (merged-usr)"),
    ("/proc", "fresh procfs for the pid namespace"),
    ("/dev", "minimal device set"),
    ("/tmp", "tmpfs, private"),
    ("/home/sandbox", "empty home"),
    ("/etc", "sealed tmpfs: alternatives only"),
    ("/workspace", "the ONLY host-writable mount"),
]
for i, (path, what) in enumerate(mounts):
    yy = 478 + i * 27
    p.box(60, yy, 150, 24, path, WHITE, 9)
    p.label(220, yy, 300, 24, what, 9, False, "left")

p.box(560, 440, 510, 240, "Workspace binding follows the profile", CONT, 12, 1)
p.box(585, 480, 460, 42, "profile.workspace_write   →   --bind  <root>  /workspace",
      CORE, 10)
p.box(585, 530, 460, 42, "profile.workspace_read    →   --ro-bind  <root>  /workspace",
      CORE, 10)
p.box(585, 580, 460, 42, "neither                             →   --dir  /workspace",
      NEUTRAL, 10)
p.note(585, 630, 460, 52,
       "/etc is a sealed read-only tmpfs holding only /etc/alternatives, "
       "so cc and awk resolve; nothing else from the host /etc is exposed.", 9)
pages.append(p)

# =========================================================================
# 5. Network attachment protocol
# =========================================================================
p = Page("network")
p.label(0, 8, 1080, 30,
        "NETWORK backend  —  timing-independent slirp4netns attachment", 16, True)

p.box(40, 55, 34, 28, "#", NEUTRAL, 10, 1)
p.box(84, 55, 160, 28, "actor", NEUTRAL, 10, 1)
p.box(254, 55, 806, 28, "step", NEUTRAL, 10, 1)

SUP, BWR, SLI = "supervisor", "bwrap", "slirp4netns"
steps = [
    ("1", SUP, "spawn bwrap with --unshare-net, passing info-fd / block-fd / sync-fd"),
    ("2", BWR, "reports JSON on info-fd:  child-pid, net-namespace inode"),
    ("3", SUP, "PIN  open(/proc/<child>/ns/net, O_RDONLY|O_CLOEXEC)"),
    ("4", SUP, "VERIFY  st_ino == reported net-namespace   else fail closed"),
    ("5", SUP, "TYPE  NS_GET_NSTYPE == CLONE_NEWNET   else fail closed"),
    ("6", SUP, "DERIVE  owner = ioctl(net_fd, NS_GET_USERNS)"),
    ("7", SUP, "TYPE  NS_GET_NSTYPE == CLONE_NEWUSER   else fail closed"),
    ("8", SUP, "open pidfd on child-pid   (process identity, a distinct role)"),
    ("9", SUP, "spawn slirp: --netns-type=path --userns-path=/proc/self/fd/<owner>"),
    ("10", SLI, "signals ready-fd; the supervisor then closes both namespace handles"),
    ("11", SUP, "RELEASE  os.write(sync_fd, b\"x\")  —  the only release primitive"),
    ("12", BWR, "COMMAND finally executes: DNS 10.0.2.3, host loopback blocked"),
]
ACTOR_FILL = {SUP: CORE, BWR: SANDBOX, SLI: MODEL}
for i, (n, actor, text) in enumerate(steps):
    yy = 90 + i * 36
    p.box(40, yy, 34, 28, n, NEUTRAL, 10, 1)
    p.box(84, yy, 160, 28, actor, ACTOR_FILL[actor], 9)
    # The pinning sequence is the part that removes the timing dependency.
    style = SANDBOX if n in ("3", "4", "5", "6", "7") else WHITE
    p.box(254, yy, 806, 28, text, style, 10)

p.note(40, 540, 1020, 100,
       "Why the pin matters — bwrap performs --unshare-user in two steps: it "
       "creates a first user namespace to build the sandbox, then a second, "
       "nested one to drop privileges. A helper that resolves /proc/<pid>/ns/user "
       "even 20 ms later joins the nested namespace and no longer has authority "
       "over the netns owner, so setns(CLONE_NEWNET) returns EPERM. Measured: "
       "a deliberate 20 ms delay failed 40 runs out of 40 with the PID-based "
       "form, and 0 out of 160 with the pinned form. The owning user namespace "
       "is derived from the netns object itself, which never changes.", 10)
pages.append(p)

# =========================================================================
# 6. cgroup resource control
# =========================================================================
p = Page("cgroup")
p.label(0, 8, 1080, 30,
        "Resource control  —  one transient systemd user scope per command",
        16, True)

p.box(40, 55, 1000, 60, "/sys/fs/cgroup/user.slice/user-<uid>.slice/"
      "user@<uid>.service/app.slice/", CONT, 11, 1)
p.label(60, 85, 960, 22,
        "delegated controllers:  cpu   memory   pids        "
        "(io and cpuset are NOT delegated — they need root)", 10, False, "left")

inside = p.box(60, 140, 520, 260, "spear-tool-<uuid>.scope   (INSIDE)",
               CONT + SANDBOX, 12, 1)
p.box(90, 185, 460, 34, "bwrap", SANDBOX, 10)
p.box(90, 225, 460, 34, "bwrap namespace child", SANDBOX, 10)
p.box(90, 265, 460, 34, "prlimit → COMMAND and every descendant", DANGER, 10)
p.label(90, 310, 460, 80,
        "memory.max · memory.swap.max · pids.max · cpu.max\n"
        "cleaned up by --collect; killed as a tree by\n"
        "systemctl --user kill --kill-whom=all --signal=KILL", 10)

outside = p.box(610, 140, 430, 260, "OUTSIDE the scope", CONT, 12, 1)
p.box(635, 185, 380, 40, "Python supervisor\n(must survive every abuse)", CORE, 10)
p.box(635, 235, 380, 40, "slirp4netns\n(never an OOM collateral victim)", MODEL, 10)
p.box(635, 285, 380, 40, "llama-server / spear-llm.service\n(system.slice)", MODEL, 10)
p.label(635, 335, 380, 55,
        "A separate scope per command means no\ncross-talk between tool calls.", 10)

p.box(40, 425, 1000, 250, "Measured behaviour of the candidate contract", CONT, 12, 1)
cells = [
    ("MemoryMax", "2 GiB", "runaway killed at exactly the cap"),
    ("MemorySwapMax", "0", "swap delta measured at 0 kB"),
    ("TasksMax", "256", "fork bomb refused with EAGAIN at 256"),
    ("CPUQuota", "800%", "throttling only — never a kill"),
]
p.box(70, 465, 230, 30, "property", NEUTRAL, 11, 1)
p.box(310, 465, 180, 30, "candidate", NEUTRAL, 11, 1)
p.box(500, 465, 510, 30, "observed enforcement", NEUTRAL, 11, 1)
for i, (a, b, c) in enumerate(cells):
    yy = 500 + i * 34
    p.box(70, yy, 230, 30, a, WHITE, 10)
    p.box(310, yy, 180, 30, b, CORE, 10)
    p.box(500, yy, 510, 30, c, WHITE, 10)
p.label(70, 640, 940, 24,
        "These values are the active production defaults, carried by "
        "CommandRunner.", 10)
pages.append(p)

# =========================================================================
# 7. Directory layout
# =========================================================================
p = Page("layout")
p.label(0, 8, 1080, 30, "/opt/llm/spear  —  directory layout", 16, True)

entries = [
    ("spear/", "9.6 G", "the SPEAR client application AND its Python venv", CORE),
    ("  rag_chat.py", "160 K", "REPL, prompt assembly, retrieval, tool dispatch", CORE),
    ("  tool_runtime.py", "80 K", "policy, workspace, sandbox, resource control, audit", CORE),
    ("  model_backend.py", "20 K", "provider-neutral model turns", CORE),
    ("  training*.py · sft_dataset.py", "—", "the fine-tuning subsystem (15 modules)", CORE),
    ("  tests/", "—", "44 modules, 851 tests", CORE),
    ("  server/inference/serve.sh", "4 K", "llama-server launcher (model/adapter resolution)", CLI),
    ("  spear-chat.sh", "7 K", "chat launcher", CLI),
    ("  active-model.conf", "—", "path of the served GGUF", NEUTRAL),
    ("  active-lora.conf", "—", "LoRA adapter, or 'none'", NEUTRAL),
    ("  pod.conf · reds.conf", "—", "remote pod, REDS host (--remote / --reds)", NEUTRAL),
    ("  projects.json", "—", "named workspaces the chat can open", NEUTRAL),
    ("  rules.d/ · skills/", "—", "prompt fragments injected per task", NEUTRAL),
    ("  rules.d/corpora/", "—", "per-corpus orientation maps", NEUTRAL),
      ("  tool-guide.md", "6 K", "tool usage guide given to the model", NEUTRAL),
    ("  chromadb/", "—", "vector store for the RAG corpus", STORE),
    ("  audit/", "—", "tool-actions.jsonl, sessions, training-data", STORE),
    ("  rules-learned.md", "—", "rules taught with /recall, every corpus", STORE),
    ("  history*.json(l)", "—", "conversation transcripts", STORE),
    ("models/gguf/", "47 G", "quantised GGUF models actually served", MODEL),
    ("models/", "75 G", "upstream HF checkpoints and conversions", MODEL),
    ("llama.cpp-next/", "1.3 G", "llama.cpp build providing llama-server", MODEL),
    ("qwen3-finetune/", "5.6 G", "trainers, pod scripts, the load preflight", NEUTRAL),
    ("src/", "351 M", "submodule: exact copy of the pod's ML packages", NEUTRAL),
    ("corpora/", "27 M", "small vendored trees indexed as own corpora", NEUTRAL),
    ("docker/", "26 K", "image, run wrapper, entrypoint, corpus registry", NEUTRAL),
    ("pod-artifacts-qwen3coder/", "111 K", "remote pod recipes and artefacts", NEUTRAL),
    ("claude/", "1.9 M", "working notes on the registered source trees", NEUTRAL),
    ("doc/", "—", "this documentation (Sphinx)", WHITE),
]
LEFT = "align=left;"
p.box(40, 50, 380, 34, "path", NEUTRAL + LEFT, 11, 1)
p.box(420, 50, 110, 34, "size", NEUTRAL, 11, 1)
p.box(530, 50, 520, 34, "content", NEUTRAL + LEFT, 11, 1)
for i, (path, size, what, style) in enumerate(entries):
    yy = 86 + i * 28
    # The two-space prefixes carry the nesting. The column must be left
    # aligned, and the indent uses non-breaking spaces because SVG renderers
    # collapse ordinary leading whitespace regardless of xml:space.
    p.box(40, yy, 380, 26, path.replace("  ", " " * 5, 1), style + LEFT, 9)
    p.box(420, yy, 110, 26, size, WHITE, 9)
    p.box(530, yy, 520, 26, what, WHITE + LEFT, 9)
pages.append(p)

# =========================================================================
# 8. Test topology
# =========================================================================
p = Page("testing")
p.label(0, 8, 1080, 30, "Test topology", 16, True)

p.box(40, 55, 1000, 180, "Always on  —  no external prerequisite", CONT, 12, 1)
p.box(70, 95, 460, 55, "tests/  —  44 modules, 851 tests\nruntime · context · sessions · training · tools", CORE, 10)
p.box(560, 95, 450, 55, "tests/test_tool_runtime.py  —  307 of them\npolicy · workspace · sandbox · cgroup · audit", CORE, 10)
p.box(70, 165, 940, 55,
      "PinnedNamespaceAttachmentTests  —  helper argv, pass_fds, FD hygiene, "
      "namespace type checks, fail-closed paths", CORE, 10)

p.box(40, 255, 1000, 300, "Opt-in  —  guarded by an environment flag", CONT, 12, 1)
flags = [
    ("SPEAR_TEST_NETWORK=1", "outbound DNS and TLS through slirp4netns",
     "needs Internet"),
    ("SPEAR_TEST_CGROUP=1", "real transient scopes: memory, pids, cpu, timeout, "
     "membership", "needs a systemd user bus with delegation"),
    ("SPEAR_TEST_NETWORK_RACE=1", "40x repetitions with a deliberate 20/50 ms delay, "
     "plus CPU load", "long running"),
]
for i, (flag, what, need) in enumerate(flags):
    yy = 300 + i * 80
    p.box(70, yy, 330, 62, flag, SANDBOX, 10, 1)
    p.box(420, yy, 400, 62, what, WHITE, 10)
    p.box(840, yy, 170, 62, need, NEUTRAL, 9)

p.note(40, 575, 1000, 110,
       "Residual-state checks belong to the suites themselves: after every run "
       "there must be zero spear-tool-*.scope unit, zero orphan bwrap or "
       "slirp4netns process, and no descriptor growth in the supervisor. "
       "The characterization test test_characterization_old_pid_based_attachment_still_races "
       "deliberately asserts that the OLD attachment still fails — if a future "
       "slirp4netns or kernel makes it safe, that test turns red and tells us "
       "the workaround can be revisited.", 10)
pages.append(p)


# ---- page: agent harness components and interactions ---------------------
# Three columns, short edges. A long arrow would be drawn straight through the
# boxes between its ends -- the same reason the overview page leaves one out --
# so cross-cutting relations live in the notes instead of in diagonals.
p = Page("agent", 900)
p.label(0, 8, 1080, 30, "Agent harness — components and their interactions", 16, True)
p.label(0, 34, 1080, 18,
        "The spine runs down the left. Everything below the application row is "
        "provider-neutral: no module there imports rag_chat.", 10)

L, M, R = 40, 385, 735
LW, MW, RW = 300, 300, 305

# row 1 — application
app  = p.box(L, 66, LW, 74, "spear-chat  ·  rag_chat.py\n\nterminal · confirmation · corpus\nregistry · slash commands", CLI, 10)
ret  = p.box(M, 66, MW, 74, "retrieval\n\nChromaDB  +  bge-m3 embedder\none collection per corpus", CLI, 10)
know = p.box(R, 66, RW, 74, "durable knowledge\n\nmemories-*.md  ·  skills/\nrules.d/  ·  system-prompt.md", CLI, 10)

# row 2 — controller
ctrl = p.box(L, 172, LW, 78, "TaskController\n\nbounded synchronous lifecycle\nfor exactly one task", CORE, 10)
child= p.box(M, 172, MW, 78, "PlanningService   (deterministic)\nExplorationService — Explorer\nReviewService — Reviewer\nboth isolated, read-only, opt-in", CORE, 10)
verif= p.box(R, 172, RW, 78, "VerificationPolicy\n\ncurrent-generation evidence only\n+ the project's own bench", CORE, 10)

# row 3 — runtime
rt   = p.box(L, 282, LW, 88, "AgentRuntime\n\nthe round loop: model turn,\nvalidation, tool calls, budgets", CORE, 10)
mb   = p.box(M, 282, MW, 88, "ModelBackend\n\nllama.cpp / OpenAI-compatible\nAnthropic\nlocal or remote endpoint", MODEL, 10)
ceng = p.box(R, 282, RW, 88, "ContextEngine + Compaction\nBudgetManager\nProgressMonitor · FailurePolicy\nCancellation", CORE, 10)

# row 4 — tools
router= p.box(L, 402, LW, 88, "ToolRouter\n+ ToolExposurePolicy\n\nschema · role · hooks · envelope\n7 tools; 2 for a read-only role", CORE, 10)
hand = p.box(M, 402, MW, 88, "registered handlers\n\nbash  ·  edit_file  ·  write_file\nappend_file  ·  remember\nsearch_corpus  ·  search_internet", CORE, 10)
wst  = p.box(R, 402, RW, 88, "WorkingState\n\ntask truth, event-reduced.\nTyped grounded events only —\nnever the conversation.", STORE, 10)

# row 5 — confinement
polic= p.box(L, 522, LW, 88, "CommandPolicy · Workspace\nToolPolicy\n\nclassify → capabilities →\nauthorize.  Fail-closed.", SANDBOX, 10)
runr = p.box(M, 522, MW, 88, "CommandRunner\n\nBubblewrap  ·  systemd scope\nslirp4netns  ·  prlimit", DANGER, 10)
evid = p.box(R, 522, RW, 88, "ResultStore — full tool output\nCheckpointManager — pre-\nmutation bytes\nAuditLogger — metadata only", STORE, 10)

# row 6 — what survives the turn
p.box(L, 636, 1000, 96, "Under STATE_DIR  —  what the session accumulates, mounted at /state in the container", CONT, 11, 1)
p.box(L + 20, 668, 300, 52, "SessionStore\nappend-only, resumable", STORE, 10)
p.box(M - 15, 668, 300, 52, "audit/runtime-trace.jsonl\nspans; never arguments", STORE, 10)
p.box(R - 10, 668, 305, 52, "trajectories.jsonl\npass · fail · unrated", STORE, 10)

# edges: vertical spine, horizontal pairs. Nothing diagonal.
p.edge(app, ctrl, "objective")
p.edge(ctrl, rt, "run")
p.edge(rt, router, "tool call")
p.edge(router, polic, "bash · mutations")
p.edge(app, ret, "query", ARR_D)
p.edge(app, know, "", ARR_D)
p.edge(ctrl, child)
p.edge(child, verif)
p.edge(rt, mb, "turns")
p.edge(ceng, mb, "composed context", ARR_D)
p.edge(router, hand)
p.edge(hand, wst, "grounded events")
p.edge(polic, runr, "granted capabilities")
p.edge(runr, evid, "evidence", ARR_D)

p.note(40, 748, 490, 132,
       "Fail-closed, everywhere. A mechanism that cannot be honoured is an "
       "error, never a downgrade: no sandbox means the command does not run "
       "unsandboxed. Since 2026-08-25 the same rule holds within a turn — a "
       "command that reports the sandbox missing makes every later mutation "
       "refuse with DENIED rather than edit what nobody can check.", 10)
p.note(550, 748, 490, 132,
       "Two loops close on evidence rather than on the model's claim. Inside "
       "the round loop, a mutation carrying no current-generation "
       "verification earns one re-prompt. Above it, a failed project bench "
       "earns exactly one bounded repair attempt, then re-verification. "
       "Whatever is still unverified is prefixed onto the answer, named file "
       "by file.", 10)
pages.append(p)

# ---- page: every component, by layer -------------------------------------
# The exhaustive map: all thirty harness modules, what they own, and what sits
# outside. The "agent" page is the readable version of the same thing; this one
# is the reference you check a name against. Line counts are read from the tree
# at generation time (see loc()).
p = Page("components", 1380)
p.label(0, 8, 1080, 30, "SPEAR — every component of the harness", 16, True)
p.label(0, 34, 1080, 18,
        "Dependency direction is strictly downward: nothing below the "
        "application band imports rag_chat, and the bottom band imports nothing "
        "of the project at all.", 10)

# Columns derived from the band bounds rather than typed in: the outer band
# runs 35..1045, so four columns inset by 20 with 12 of gap end inside the
# frame instead of one pixel past it -- which is what typing them did.
def columns(left, right, count=4, inset=20, gap=12):
    x0, x1 = left + inset, right - inset
    w = (x1 - x0 - gap * (count - 1)) // count
    return [x0 + i * (w + gap) for i in range(count)], w


COLS, CW = columns(35, 1045)


def row(y, items, style=CORE, h=52, fs=9, cols=None, width=None):
    xs, w = (cols, width) if cols else (COLS, CW)
    return [p.box(x, y, w, h, text, style, fs)
            for x, text in zip(xs, items) if text]


# ── application ──────────────────────────────────────────────────────────
p.box(35, 60, 1010, 148, "Application  —  the only layer that talks to a human", CONT, 11, 1)
row(92, ["spear-chat.sh\nREPL launcher, backend flags",
         "scripts/docker/spear-docker.sh\ncontainer, /state, /corpora",
         loc("backend_select") + "\nstartup backend picker",
         loc("rag_chat") + "\nsession · corpora · UI · handlers"], CLI)
row(150, [loc("index_corpus") + "\ncurated build-system walk",
          loc("index_dir") + "\ngeneric tree indexer",
          loc("embedding") + "\nbge-m3, one definition",
          loc("memory_store") + "\nmemories-*.md + sidecar"], CLI)

# ── orchestration ────────────────────────────────────────────────────────
p.box(35, 224, 1010, 148, "Orchestration  —  one task, bounded and synchronous", CONT, 11, 1)
row(256, [loc("task_controller") + "\nlifecycle, gates, warnings",
          loc("planning") + "\ndeterministic plan steps",
          loc("orchestration") + "\nExplorer delegation (opt-in)",
          loc("reviewer") + "\nindependent review (opt-in)"])
row(314, [loc("agent_roles") + "\nmain · explorer · reviewer · planning",
          loc("tool_exposure") + "\nrole-aware registry view",
          loc("verification") + "\ncurrent-generation evidence",
          loc("diff_evidence") + "\nbounded change evidence"])

# ── runtime ──────────────────────────────────────────────────────────────
p.box(35, 388, 1010, 148, "Runtime core  —  the model loop and everything that bounds it", CONT, 11, 1)
row(420, [loc("agent_runtime") + "\nrounds · validation · tool calls",
          loc("model_backend") + "\nprovider boundary, ModelTurn",
          loc("context_engine") + "\nlayered composition + budget",
          loc("compaction") + "\ntransactional summarisation"])
row(478, [loc("budgets") + "\nturns · calls · tokens · time",
          loc("progress_monitor") + "\naction fingerprints, stalls",
          loc("failure_policy") + "\nclassify, bounded retry",
          loc("cancellation") + " · " + loc("hooks")])

# ── truth and evidence ───────────────────────────────────────────────────
p.box(35, 552, 1010, 148, "Task truth and evidence  —  what a session leaves behind", CONT, 11, 1)
row(584, [loc("working_state") + "\ntyped events, reduced to truth",
          loc("session_store") + "\nappend-only, resumable",
          loc("result_store") + "\nfull output, out of context",
          loc("checkpoint") + "\npre-mutation bytes"], STORE)
row(642, [loc("tracing") + "\nspans, never arguments",
          "trajectories.jsonl\npass · fail · unrated",
          "history*.json · archive\nconversation + search",
          "memories-*.md · skills/\ndurable knowledge"], STORE)

# ── tools and security ───────────────────────────────────────────────────
p.box(35, 716, 1010, 264, "Tool lifecycle and security substrate", CONT, 11, 1)
row(748, [loc("tool_registry") + "\n7 specs, schemas, roles",
          loc("tool_router") + "\nvalidate · span · envelope",
          "registered handlers (rag_chat)\nbash · edit · write · append",
          "remember · search_corpus\nsearch_internet"])
p.box(COLS[0], 810, CW * 4 + 36, 156, loc("tool_runtime") + "   —   dependency-free safety primitives", CONT, 10, 1)
SUB, SUBW = columns(COLS[0], COLS[0] + CW * 4 + 36, inset=14, gap=10)
row(842, ["CapabilityPolicy · ExecutionMode\nSAFE · ASK · AUTO",
          "CommandPolicy\nclassify → authorize",
          "Workspace · ToolPolicy\ncanonical root, no escape",
          "ExecutionProfile\ncontract from capabilities"], SANDBOX, 48,
    cols=SUB, width=SUBW)
row(896, ["CommandRunner\nthe execution boundary",
          "BubblewrapSandbox\nargv, preflight, pidfd",
          "SystemdScopeRunner\nResourceLimits · CgroupLimits",
          "AuditLogger\nappend-only, metadata only"], SANDBOX, 48,
    cols=SUB, width=SUBW)

# ── training ─────────────────────────────────────────────────────────────
# Fed by the turn, never called from it: nothing in this band is model-visible
# and freezing a bundle runs no training. It is drawn below the security
# substrate rather than beside the runtime for that reason.
p.box(35, 996, 1010, 148,
      "Training data as a by-product  —  operator-only, never model-visible", CONT, 11, 1)
row(1028, [loc("training_data") + " · " + loc("training_store") + "\nFT0 canonical capture",
           loc("sft_dataset") + " · " + loc("training_export") + "\nFT1 one target per sample",
           loc("preference_dataset") + "\nFT2 paired · unpaired",
           loc("training_bundle") + "\nFT3 frozen, checksummed"], STORE)
row(1086, [loc("training_readiness") + "\nstate · level · strategy",
           loc("training_governance") + " · " + loc("training_splits") + "\norigins, holdout, hashed",
           loc("training_controller") + " · " + loc("training_jobs") + "\nFT4 operator plane",
           loc("training_handoff") + " · " + loc("inference_service") + "\nthe card that serves trains"], STORE)

# ── outside ──────────────────────────────────────────────────────────────
p.box(35, 1160, 1010, 92, "Outside the harness", CONT, 11, 1)
row(1192, ["llama-server (llama.cpp-next)\nthe served model",
           "Anthropic API\noptional backend",
           "ChromaDB\none collection per corpus",
           "the host: bwrap · systemd\nslirp4netns · the commands"], DANGER, 48)

p.note(35, 1268, 1010, 92,
       "Read the bands as ownership, not as call order. WorkingState is task "
       "truth and changes only through typed grounded events; SessionStore is "
       "resumable runtime state; ResultStore owns large evidence; "
       "CheckpointManager owns recovery; MemoryStore owns durable knowledge; "
       "ContextEngine is the only production context composer. A module in a "
       "lower band never reaches up: the application constructs, the runtime "
       "executes, the substrate confines.", 10)
pages.append(p)

# ---- page: the same overview, drawn from the code ------------------------
# Same shape and same legend as the hand-made picture in _static, built here so
# every box can be checked against the tree. What the drawn one got wrong:
# TaskController appeared in two bands, and RetryPolicy was drawn beside
# FailurePolicy although it lives inside it.
p = Page("architecture_dark", 970, background=INK)
p.label(0, 10, 1080, 30, "SPEAR agent harness — architecture",
        17, True, color="#e6f6ff")

def band(y, h, title, sub=""):
    p.box(24, y, 1052, h, "", D_BAND, 10)
    p.label(38, y + 8, 190, 20, title, 11, True, align="left",
            color="#e6f6ff")
    if sub:
        p.label(38, y + 26, 190, 18, sub, 9, False, align="left",
                color="#8fb3d9")


def cells(y, items, style, h=46, x0=240, right=1064, gap=10, fs=9):
    n = len(items)
    w = (right - x0 - gap * (n - 1)) // n
    return [p.box(x0 + i * (w + gap), y, w, h, text, style, fs)
            for i, text in enumerate(items)]


# legend
leg = [("CORE / default on", D_CORE), ("optional", D_OPT),
       ("experimental / default off", D_EXP), ("security boundary", D_SEC)]
for i, (text, style) in enumerate(leg):
    p.box(470 + i * 152, 44, 146, 24, text, style, 8)

# ── application ──────────────────────────────────────────────────────────
band(80, 74, "Application", "the only human-facing layer")
cells(96, ["spear-chat  ·  spear-docker\nterminal, flags, confirmation",
           "rag_chat.py\nsession · corpora · retrieval · handlers",
           "memories-*.md  ·  skills/  ·  rules.d/\ndurable knowledge"], D_CORE)

# ── orchestration ────────────────────────────────────────────────────────
band(166, 106, "Orchestration", "one task, bounded")
cells(182, ["TaskController\nthe single lifecycle owner",
            "PlanningService\ndeterministic, conservative",
            "VerificationPolicy\ncurrent-generation evidence"], D_CORE, 44)
cells(234, ["ExplorationService — Explorer\nisolated, read-only  (opt-in)",
            "ReviewService — Reviewer\nisolated, read-only  (opt-in)",
            "review repair loop\none bounded attempt  (opt-in)"], D_EXP, 30)

# ── runtime ──────────────────────────────────────────────────────────────
band(284, 112, "Core runtime", "provider-neutral")
cells(300, ["AgentRuntime\nrounds · validation · tool calls",
            "ModelBackend\nllama.cpp / OpenAI  ·  Anthropic",
            "ContextEngine  +  CompactionService\nthe only context composer"], D_CORE, 44)
cells(352, ["BudgetManager", "ProgressMonitor",
            "FailurePolicy\n(RetryPolicy lives inside it)",
            "CancellationToken", "hooks"], D_CORE, 30, fs=8)

# ── evidence ─────────────────────────────────────────────────────────────
band(408, 106, "Evidence & persistence", "under STATE_DIR")
cells(424, ["WorkingState\ntask truth, typed events",
            "SessionStore\nappend-only, resumable",
            "ResultStore\nlarge tool evidence"], D_CORE, 44)
cells(476, ["CheckpointManager\npre-mutation bytes", "tracing\nspans",
            "trajectories.jsonl\npass · fail · unrated",
            "history · archive"], D_OPT, 30, fs=8)

# ── tools ────────────────────────────────────────────────────────────────
band(518, 150, "Tool & security", "no CLI dependency")
cells(534, ["ToolRegistry\n7 specs, role-filtered views",
            "ToolExposurePolicy\nwhat this role may see",
            "ToolRouter\nvalidate · span · envelope"], D_CORE, 44)
cells(586, ["registered handlers\nbash · edit · write · append · remember · search",
            "HookManager\nobserver only"], D_CORE, 30, fs=8)
p.box(240, 624, 824, 34,
      "CapabilityPolicy  ·  CommandPolicy  ·  Workspace  ·  ToolPolicy  ·  ExecutionProfile"
      "      →      CommandRunner  ·  Bubblewrap  ·  systemd scope  ·  AuditLogger",
      D_SEC, 9)

# ── outside ──────────────────────────────────────────────────────────────
band(680, 74, "Outside", "not ours")
cells(696, ["llama-server (llama.cpp-next)\nthe served model",
            "ChromaDB  +  bge-m3\none collection per corpus",
            "the host\nbwrap · systemd · slirp4netns"], D_OUT)

p.note(24, 778, 1052, 74,
       "Two rules the picture cannot show. Dependency runs one way: no module "
       "below the application band imports rag_chat, and tool_runtime imports "
       "nothing of the project at all. And every gate is fail-closed — a "
       "mechanism that cannot be honoured is an error, never a quiet "
       "downgrade, including within a turn: once a command reports the sandbox "
       "missing, every later mutation is refused.", 10,
       style="fillColor=#101a33;strokeColor=#334166;fontColor=#cfe3ff;")
p.note(24, 862, 1052, 84,
       "Against the hand-made overview in _static: TaskController is one box "
       "here, not two; RetryPolicy is named inside FailurePolicy, where it "
       "lives; the Explorer and Reviewer boxes carry the same experimental "
       "colour as their repair loop; and the security band is the whole "
       "authorization chain rather than two boxes, because nothing reaches "
       "CommandRunner without crossing all of it.", 10,
       style="fillColor=#101a33;strokeColor=#334166;fontColor=#cfe3ff;")
pages.append(p)

# =========================================================================
# The engineering workflow — INVESTIGATE, PLAN, EDIT, TEST, REVIEW
# =========================================================================
# One page per concept the documentation explains, so the picture and the
# prose cannot drift: this is the same five stages workflow.rst describes,
# with the condition that opens each gate written on the arrow.
p = Page("workflow", 660)
p.label(0, 8, 1080, 30, "The engineering workflow", 16, True)
p.label(0, 34, 1080, 18,
        "Each stage opens only on a condition the runtime can check. A stage "
        "that cannot open says so; it does not proceed on an assumption.", 10)

STAGE_W, STAGE_H, STAGE_Y = 186, 92, 96
xs = [30 + i * 212 for i in range(5)]

inv = p.box(xs[0], STAGE_Y, STAGE_W, STAGE_H,
            "INVESTIGATE\n\nread the authoritative\nsource and the\nimplementation",
            CORE, 10, 1)
pln = p.box(xs[1], STAGE_Y, STAGE_W, STAGE_H,
            "PLAN\n\none recorded item per\nrequirement, each with\nits validation",
            CORE, 10, 1)
edt = p.box(xs[2], STAGE_Y, STAGE_W, STAGE_H,
            "EDIT\n\nwrite only the files a\nplanned item named",
            CORE, 10, 1)
tst = p.box(xs[3], STAGE_Y, STAGE_W, STAGE_H,
            "TEST\n\nrun the validation that\nreaches the changed\nbehaviour",
            CORE, 10, 1)
rev = p.box(xs[4], STAGE_Y, STAGE_W, STAGE_H,
            "REVIEW\n\nreport per requirement\nfrom the ledger",
            CORE, 10, 1)

p.edge(inv, pln, "both kinds of\nevidence held")
p.edge(pln, edt, "item accepted")
p.edge(edt, tst, "file written")
p.edge(tst, rev, "result recorded")

# The gate and the ledger sit under the stages they govern.
gate = p.box(xs[1], 236, STAGE_W * 2 + 26, 62,
             "write gate — a file is writable only because a planned item "
             "named it", SANDBOX, 10)
p.edge(pln, gate, "", ARR_D)
p.edge(gate, edt, "", ARR_D)

led = p.box(30, 330, 1020, 62,
            "requirement ledger — one row per requirement: its evidence, its "
            "disposition, its validation.\nThe closing report is read from "
            "this, not from the answer's prose.", STORE, 10)
p.edge(inv, led, "", ARR_D)
p.edge(rev, led, "", ARR_D)

p.box(30, 420, 1020, 56,
      "replan — a validation that keeps failing the same way reopens PLAN "
      "rather than being patched again", NEUTRAL, 10)

p.note(30, 496, 1020, 110,
       "Why the stages are separate rather than one pass:\n"
       "  \u2022 a requirement nobody read cannot be planned against, so "
       "INVESTIGATE holds both the authoritative and the implementation side "
       "before PLAN opens;\n"
       "  \u2022 deciding how a behaviour will be proved BEFORE writing it is "
       "what makes TEST a check rather than a description, so a plan item "
       "carries its validation;\n"
       "  \u2022 the ledger, not the closing prose, decides whether the turn "
       "is finished \u2014 which is how \u2018incomplete\u2019 and "
       "\u2018out of scope\u2019 survive to the report.", 10)
pages.append(p)

# =========================================================================
# Authoritative evidence — where a normative claim is allowed to come from
# =========================================================================
p = Page("evidence", 700)
p.label(0, 8, 1080, 30, "Authoritative evidence, and what may rest on it", 16, True)
p.label(0, 34, 1080, 18,
        "Two sources, two roles, and one rule: the document is the only "
        "source of normative force.", 10)

q = p.box(390, 66, 300, 46, "the question asked this turn", CLI, 10)

sc = p.box(390, 138, 300, 50,
           "scope\nNORMATIVE \u00b7 IMPLEMENTATION \u00b7 MIXED", CORE, 10)
p.edge(q, sc)

# left: authority. right: implementation.
p.box(30, 214, 490, 250, "Normative authority", CONT, 12, 1)
bind = p.box(55, 254, 440, 44,
             "bound standard \u2014 one document and revision, per machine",
             CORE, 10)
stool = p.box(55, 308, 440, 52,
              "standard.search \u00b7 standard.fetch\n"
              "standard.get_structure \u00b7 standard.cite", CORE, 10)
prov = p.box(55, 370, 440, 76,
             "provision records\n"
             "identity (Rule / Permission / Observation, ordinal)\n"
             "modality \u00b7 section \u00b7 page \u00b7 source id", STORE, 10)

p.box(560, 214, 490, 250, "Implementation evidence", CONT, 12, 1)
ctool = p.box(585, 254, 440, 44,
              "bash \u00b7 search_corpus \u2014 withheld on a normative turn",
              NEUTRAL, 10)
cread = p.box(585, 308, 440, 52,
              "what a tool actually RETURNED this turn\n"
              "(not what a file somewhere happens to contain)", NEUTRAL, 10)
cgnd = p.box(585, 370, 440, 76,
             "grounds EXISTENCE only\n"
             "a name in a comment is still just a name:\n"
             "no modality, no authority, no clause", DANGER, 10)

p.edge(sc, bind, "authoritative first")
p.edge(bind, stool)
p.edge(stool, prov)
p.edge(sc, ctool, "implementation scope", ARR_D)
p.edge(ctool, cread)
p.edge(cread, cgnd)

ans = p.box(300, 492, 480, 52,
            "the answer \u2014 every normative claim carries a citation",
            WHITE, 11, 1)
p.edge(prov, ans, "may establish\nwhat is REQUIRED")
p.edge(cgnd, ans, "may illustrate\nwhat the code DOES", ARR_D)

p.note(30, 568, 1020, 110,
       "The guards read the two ledgers separately, and that is the whole "
       "point:\n"
       "  \u2022 a claim may not be stronger than the provision it cites "
       "(informative < may < should < shall);\n"
       "  \u2022 a technical name credited to the document must occur in the "
       "document;\n"
       "  \u2022 a count, a maximum or a minimum needs a clause that states "
       "one \u2014 counting flags in an implementation does not;\n"
       "  \u2022 where nothing supports the claim, the answer is withheld "
       "with the reason, rather than issued with a guess.", 10)
pages.append(p)

# ---- page: the hand-made overview, pasted in ------------------------------
# A picture that did not come from this generator: it is carried as a page so
# the .drawio stays the one file holding every diagram of this documentation,
# and so it can be annotated beside the generated ones. Skipped when the file
# is absent, exactly like loc() degrades — a fresh checkout must still generate.
_FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "_static", "spear-harness-final-architecture.png")
if os.path.exists(_FINAL):
    p = Page("final_architecture", 1000)
    p.label(0, 8, 1080, 30, "Agent harness — final architecture (overview)", 16, True)
    p.label(0, 34, 1080, 18,
            "Hand-made overview, kept as pasted. The generated pages are the "
            "ones checked against the tree.", 10)
    # 1672 x 941 in the file; scaled to the page width, aspect preserved.
    p.image(20, 66, 1060, 597, _FINAL)
    p.note(20, 690, 1060, 96,
           "Read this one for the shape of the thing, not for the details: "
           "TaskController appears in two bands, and FailurePolicy and "
           "RetryPolicy are drawn as peers although the second lives inside "
           "the first (failure_policy.py). The 'components' page carries the "
           "same map with the module names and line counts read from the "
           "tree.", 10)
    pages.append(p)

# ---- emit ----------------------------------------------------------------
out = ('<mxfile host="spear-doc" type="device">'
       + "".join(pg.xml() for pg in pages) + "</mxfile>")
with open("spear.drawio", "w") as fh:
    fh.write(out)
print(f"wrote spear.drawio with {len(pages)} pages:")
for i, pg in enumerate(pages):
    if pg.raster_only:
        print(f"  [{i}] {pg.name:10s} -> (drawio page only, embedded raster)")
        continue
    name = f"spear_{pg.name}.svg"
    with open(name, "w") as fh:
        fh.write(pg.svg())
    print(f"  [{i}] {pg.name:10s} -> {name}")
