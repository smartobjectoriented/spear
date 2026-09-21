# SPEAR visual identity

The mark is **concept C**: three stages narrowing into one result.

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
| `spear-logo-horizontal.svg` | `html_logo` — the sidebar lockup, mark + SPEAR |
| `spear-logo.svg` | the landing-page lockup, larger mark + SPEAR |
| `spear-mark.svg` | the mark alone, for anywhere it is shown large |
| `spear-mark-small.svg` | the mark alone below ~24 px |
| `spear-icon.svg` | `html_favicon` — the small mark, square, with margin |

Two variants of one mark, not two marks. The small one drops the
specification's slot, because below about 24 px that slot is a sub-pixel line
that only muddies the bar it sits in. Everything else — proportions, colours,
corner radii — is identical, so the funnel is the same object at every size.

The wordmark is monoline geometry, not live text. A logo that renders
differently on a machine without the right font is not a logo, so **no asset
here references a font**; the landing lockup carries no expansion line either,
because the page prints the full name as its own title immediately below it.

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
python3 gen_identity.py    # rewrites the five SVGs in doc/source/img
python3 proof.py           # renders proof.png: 16/24/32/48/64 px, sidebar,
                           # light card, dark, monochrome
```

`gen_identity.py` is the single description. The mark's box and the wordmark's
cap height are constants there, and the lockups compute their alignment from
them — a hand-placed offset goes wrong silently the first time either moves.
`proof.png` itself is disposable and is not committed.

## Where it is wired in

- `doc/source/conf.py` — `html_logo`, `html_favicon`
- `doc/source/index.rst` — the landing lockup, as a block above the title
- `doc/source/_static/theme_overrides.css` — the light card behind the sidebar
  logo (the theme's search area is a medium blue and the accent sits close to
  it), suppression of the duplicate project name under it, and the rule that
  keeps the landing lockup a block so the `h1` never wraps around it
