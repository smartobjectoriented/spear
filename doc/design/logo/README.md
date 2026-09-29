# SPEAR visual identity

The mark is three stages narrowing into one result.

```
specification   the authoritative source        warm, widest
     ↓
reasoning       the evidence weighed            accent, narrower
     ↓
action          exactly one engineering result  accent, a single point
```

The silhouette is a funnel, and the two colours carry that meaning rather than
decorating it: **warm is the authoritative source, accent is everything
derived from it**. That is the distinction the documentation is built on, so
the mark states it too.

## Assets

Canonical assets live in `doc/source/img/`. They are what the build consumes;
this directory holds only the source that generates them.

| Asset | Role |
|---|---|
| `spear-mark.svg` | the mark alone: `html_logo` (sidebar) and `html_favicon` |
| `spear-logo.svg` | the landing-page lockup, mark + SPEAR |
| `spear-logo-horizontal.svg` | mark + SPEAR, for use outside the manual |

The mark is drawn on a 4-unit grid inside its 64 box, so every edge falls on a
whole pixel at 16, 32, 48 and 64 px: at 16 px each stage is 4 px tall and the
gaps between them exactly 1 px. The three stages keep one height, so the
result reads as the end of a funnel rather than as an arrowhead.

One mark, with one difference by size: the landing lockup cuts a line of text
into the specification, where it is about 6 px tall. Below about 48 px that
line is sub-pixel and only muddies the bar, so the mark used everywhere else
goes without it. Proportions, colours and corner radii are identical.

The wordmark is monoline geometry, not live text. A logo that renders
differently on a machine without the right font is not a logo, so **no asset
here references a font**; the landing lockup carries no expansion line either,
because the page prints the full name as its title immediately below it.

## Palette

| Token | Value | Use |
|---|---|---|
| ink | `#12263A` | wordmark, monochrome foreground |
| accent | `#0E7C9B` | reasoning and the derived result |
| warm | `#D98A2B` | the authoritative specification |
| subtitle | `#4A5A6A` | prose set beside the mark, never inside it |

No gradients: recognisability has to survive monochrome, and it does.

The institutional HEIG-VD/REDS logotype (`img/REDS-HEIG-VD.png`) and its red
are institutional artwork. They are never restyled, recoloured or reused as
part of a project mark.

## Maintaining it

```sh
python3 gen_identity.py    # rewrites the three SVGs in doc/source/img
python3 proof.py           # renders proof.png: 16/24/32/48/64 px, sidebar,
                           # light card, dark, monochrome (not committed)
```

`gen_identity.py` is the single description. The mark's box and the wordmark's
cap height are constants there, and the lockups compute their alignment from
them — a hand-placed offset goes wrong silently the first time either moves.

## Where it is wired in

- `doc/source/conf.py`: `html_logo` and `html_favicon`, both `spear-mark.svg`.
- `doc/source/_static/theme_overrides.css`: the light card behind the sidebar
  mark, the same treatment the sibling projects give theirs, because teal on
  the theme's blue has too little contrast.
- `doc/source/index.rst`: the landing lockup as a centred block above the
  title — never floated, which wraps the title around it — then the full
  name as the title, then the institutional logotype.
