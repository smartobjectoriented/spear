#!/usr/bin/env python3
"""The SPEAR identity, emitted from one description.

Concept C: three stages narrowing into one result — the specification is
read, the evidence is weighed, and exactly one action comes out of it. The
silhouette is a funnel, wide at the source and single at the result, and the
two colours carry that meaning rather than decorating it: warm is the
authoritative source, accent is everything derived from it.

    python3 gen_identity.py        # writes the canonical SVGs into ../../source/img

Everything is drawn here rather than typed into the files, because the mark
and the wordmark share a stroke weight and a corner radius and those have to
move together. The wordmark is monoline geometry, not live text: a logo that
renders differently where a font is missing is not a logo.
"""
import pathlib

OUT = (pathlib.Path(__file__).resolve().parent / ".." / ".." / "source" / "img").resolve()

INK      = "#12263A"
ACCENT   = "#0E7C9B"
WARM     = "#D98A2B"
SUBTITLE = "#4A5A6A"   # reserved for prose beside the mark, not used in it

# ---------------------------------------------------------------- the mark
# One geometry at every size. The specification's slot is CUT OUT rather than
# painted white: a white notch is only invisible until the background stops
# being white, which on the theme's blue sidebar is immediately.
MARK = f'''  <!-- 1. the specification: the authoritative source -->
  <path fill="{WARM}" fill-rule="evenodd"
        d="M10.5 6 H53.5 A3.5 3.5 0 0 1 57 9.5 V18.5 A3.5 3.5 0 0 1 53.5 22
           H10.5 A3.5 3.5 0 0 1 7 18.5 V9.5 A3.5 3.5 0 0 1 10.5 6 Z
           M15 11 H36 A3 3 0 0 1 36 17 H15 A3 3 0 0 1 15 11 Z"/>
  <!-- 2. the evidence weighed -->
  <rect x="16" y="26" width="32" height="14" rx="3.2" fill="{ACCENT}"/>
  <!-- 3. the one action -->
  <path fill="{ACCENT}" d="M21 44 H43 L32 58 Z"/>'''

# The same three stages with the slot dropped: below about 24 px it is a
# sub-pixel line that only muddies the bar it sits in. What survives is the
# funnel, which is the whole idea.
MARK_SMALL = f'''  <rect x="7" y="6" width="50" height="16" rx="3.6" fill="{WARM}"/>
  <rect x="16" y="26" width="32" height="14" rx="3.2" fill="{ACCENT}"/>
  <path fill="{ACCENT}" d="M21 44 H43 L32 58 Z"/>'''

# ------------------------------------------------------------ the wordmark
# Monoline geometric capitals: one stroke weight, round caps and joins, the
# same language as the mark. Drawn on a 70-unit cap height with the baseline
# at y=80, so a letter's box is 10..80.
# what the mark and the wordmark actually occupy, for the lockup arithmetic
MARK_TOP, MARK_H, MARK_W = 6, 52, 64      # y 6..58 inside the 64 box
CAP_TOP, CAP_H = 10, 70                   # the wordmark's cap band

SW = 11          # stroke weight
LETTERS = {
    # (advance, path) — each path is drawn from its own x origin
    "S": (48, "M40 23 A17.5 17.5 0 1 0 22.5 45 A17.5 17.5 0 1 1 5 67"),
    "P": (44, "M5.5 80 V10 H26 A17.5 17.5 0 0 1 26 45 H5.5"),
    "E": (42, "M40 10 H5.5 V80 H40 M5.5 45 H34"),
    "A": (48, "M5.5 80 L24 10 L42.5 80 M13 56 H35"),
    "R": (46, "M5.5 80 V10 H25 A16 16 0 0 1 25 42 H5.5 M24 42 L41 80"),
}
TRACK = 13       # letter spacing


def wordmark(fill, x=0, y=0, scale=1.0):
    """SPEAR, as paths. Returns (svg fragment, advance width)."""
    parts, cursor = [], 0.0
    for ch in "SPEAR":
        adv, d = LETTERS[ch]
        parts.append(f'<path d="{d}" transform="translate({cursor:g} 0)"/>')
        cursor += adv + TRACK
    cursor -= TRACK
    body = "\n      ".join(parts)
    frag = (f'  <g transform="translate({x:g} {y:g}) scale({scale:g})"\n'
            f'     fill="none" stroke="{fill}" stroke-width="{SW}"\n'
            f'     stroke-linecap="round" stroke-linejoin="round">\n'
            f'      {body}\n  </g>')
    return frag, cursor * scale


def svg(view_w, view_h, body, label, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:g} {view_h:g}"\n'
            f'     width="{view_w:g}" height="{view_h:g}" role="img" aria-label="{label}">\n'
            f'  <title>{title}</title>\n{body}\n</svg>\n')


NAME = "SPEAR"
FULL = "SPEAR — Specification-driven Platform for Embedded Agentic Reasoning"
SUB = "Specification-driven Platform for Embedded Agentic Reasoning"


def main():
    written = []

    def write(name, text):
        (OUT / name).write_text(text)
        written.append(name)

    # the detailed mark, for anywhere it is shown large
    write("spear-mark.svg", svg(64, 64, MARK, f"{NAME} mark", FULL))

    # the small mark: sidebar, tab, favicon
    write("spear-mark-small.svg",
          svg(64, 64, MARK_SMALL, f"{NAME} mark", FULL))

    # square favicon variant: same small mark, breathing room, on nothing
    fav = ('  <g transform="translate(6 6) scale(0.8125)">\n'
           + "\n".join("  " + l for l in MARK_SMALL.splitlines()) + "\n  </g>")
    write("spear-icon.svg", svg(64, 64, fav, f"{NAME} icon", FULL))

    # The two lockups, with the mark and the wordmark optically centred on
    # one line. Computed rather than typed: the mark's box and the cap height
    # are the two numbers that move when the geometry is touched, and a
    # hand-placed translate goes wrong silently the first time either does.
    #
    #   the mark occupies y 6..58 of its 64 box      -> MARK_TOP, MARK_H
    #   the wordmark occupies cap 10..80             -> CAP_TOP, CAP_H
    def lockup(name, mark_body, mark_scale, word_scale, height, gap, pad):
        mark_h = MARK_H * mark_scale
        mark_y = (height - mark_h) / 2 - MARK_TOP * mark_scale
        cap_h = CAP_H * word_scale
        word_y = (height - cap_h) / 2 - CAP_TOP * word_scale
        word_x = pad + MARK_W * mark_scale + gap
        word, w = wordmark(INK, x=word_x, y=word_y, scale=word_scale)
        mark = (f'  <g transform="translate({pad:g} {mark_y:g}) scale({mark_scale:g})">\n'
                + "\n".join("  " + l for l in mark_body.splitlines()) + "\n  </g>")
        write(name, svg(round(word_x + w + pad), height, mark + "\n" + word,
                        NAME if "horizontal" in name else FULL, FULL))

    # the sidebar lockup takes the small mark: it is shown at about 40 px
    lockup("spear-logo-horizontal.svg", MARK_SMALL, 0.92, 0.50,
           height=64, gap=16, pad=6)

    # the landing lockup takes the detailed one, and no expansion line: the
    # page prints the full name as its own title just below, and that line
    # was the last thing here that needed a font to be installed.
    lockup("spear-logo.svg", MARK, 1.45, 0.72, height=104, gap=26, pad=8)


    for name in written:
        print(f"  wrote source/img/{name}  ({(OUT / name).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
