#!/usr/bin/env python3
"""The SPEAR identity, emitted from one description.

The mark: three stages narrowing into one result — the specification is
read, the evidence is weighed, and exactly one action comes out of it. The
silhouette is a funnel, wide at the source and single at the result, and the
two colours carry that meaning rather than decorating it: warm is the
authoritative source, accent is everything derived from it.

    python3 gen_identity.py        # writes the three canonical SVGs into ../../source/img

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
# Drawn on a 4-unit grid inside the 64 box, so every edge lands on a whole
# pixel at 16, 32, 48 and 64 px: at 16 px each stage is 4 px tall and the gaps
# between them are exactly 1 px, instead of the smeared half-pixels an
# off-grid funnel gives. The three stages keep one height, so the result reads
# as the last step of a funnel and not as an arrowhead.
#
#   specification   x 4..60   y  4..20   warm
#   reasoning       x 16..48  y 24..40   accent
#   action          x 20..44  y 44..60   accent, to a single point
MARK = f'''  <!-- 1. the specification: the authoritative source -->
  <rect x="4" y="4" width="56" height="16" rx="4" fill="{WARM}"/>
  <!-- 2. the evidence weighed -->
  <rect x="16" y="24" width="32" height="16" rx="4" fill="{ACCENT}"/>
  <!-- 3. the one action -->
  <path fill="{ACCENT}" d="M20 44 H44 L32 60 Z"/>'''

# The landing lockup shows the mark at about 90 px, where the specification
# can carry its line of text. It is CUT OUT rather than painted white -- a
# white notch is only invisible until the background stops being white -- and
# it is the only difference: below about 48 px it is a sub-pixel line, so the
# compact mark, the sidebar and the favicon go without it.
MARK_LANDING = MARK.replace(
    f'<rect x="4" y="4" width="56" height="16" rx="4" fill="{WARM}"/>',
    f'<path fill="{WARM}" fill-rule="evenodd"\n'
    '        d="M8 4 H56 A4 4 0 0 1 60 8 V16 A4 4 0 0 1 56 20 H8 A4 4 0 0 1 4 16\n'
    '           V8 A4 4 0 0 1 8 4 Z M14 10 H34 A2 2 0 0 1 34 14 H14 A2 2 0 0 1 14 10 Z"/>')

# ------------------------------------------------------------ the wordmark
# Monoline geometric capitals: one stroke weight, round caps and joins, the
# same language as the mark. Drawn on a 70-unit cap height with the baseline
# at y=80, so a letter's box is 10..80.
# what the mark and the wordmark actually occupy, for the lockup arithmetic
MARK_TOP, MARK_H, MARK_W = 4, 56, 64      # y 4..60 inside the 64 box
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

    # the mark: sidebar (html_logo), favicon, and anywhere it stands alone
    write("spear-mark.svg", svg(64, 64, MARK, f"{NAME} mark", FULL))

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

    # mark + SPEAR, for use outside the manual (README, slides, banners)
    lockup("spear-logo-horizontal.svg", MARK, 0.92, 0.50,
           height=64, gap=16, pad=6)

    # the landing lockup takes the detailed one, and no expansion line: the
    # page prints the full name as its own title just below, and that line
    # was the last thing here that needed a font to be installed.
    lockup("spear-logo.svg", MARK_LANDING, 1.45, 0.72, height=104, gap=26, pad=8)


    for name in written:
        print(f"  wrote source/img/{name}  ({(OUT / name).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
