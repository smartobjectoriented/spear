#!/bin/bash
# Render every page of spear.drawio to a PNG (spear_<name>.png).
#
#   ./export_png.sh            # every page
#   ./export_png.sh agent      # one page, by name
#
# Two ways in, tried in that order:
#
#   1. the drawio CLI, which reads spear.drawio directly. It is an Electron app,
#      so it needs a virtual X server, and the snap is confined and cannot read
#      /opt -- hence the staging directory. It is also frequently broken: 30.4.1
#      fails on EVERY input with "ReferenceError: next is not defined" from its
#      own electron.js, which is why the SVGs are rendered by the generator in
#      the first place.
#   2. rasterising the generated SVG, which needs only rsvg-convert or
#      ImageMagick and works whenever the documentation itself builds.
#
# The page list is read from the generator, so a page added there is exported
# here without editing this file -- the previous hard-coded list silently
# stopped at eight when there were ten.
set -e
cd "$(dirname "$0")"
here="$PWD"
density="${SPEAR_PNG_DENSITY:-130}"

mapfile -t names < <(sed -n 's/^p = Page("\([a-z_]*\)".*/\1/p' gen_spear_diagrams.py)
if [ $# -gt 0 ]; then names=("$@"); fi
[ ${#names[@]} -gt 0 ] || { echo "no pages found in gen_spear_diagrams.py" >&2; exit 1; }

stage=""
if command -v drawio >/dev/null 2>&1 && command -v xvfb-run >/dev/null 2>&1; then
    stage="$(mktemp -d "${HOME}/.cache/spear-drawio-XXXXXX")"
    trap 'rm -rf "$stage"' EXIT
    cp spear.drawio "$stage/"
fi

page_index() {   # drawio's -p selector is 0-based over the file's page order
    local want="$1" i=0 name
    while read -r name; do
        [ "$name" = "$want" ] && { echo "$i"; return 0; }
        i=$((i + 1))
    done < <(sed -n 's/^p = Page("\([a-z_]*\)".*/\1/p' gen_spear_diagrams.py)
    return 1
}

for name in "${names[@]}"; do
    out="$here/spear_${name}.png"
    rm -f "$out"

    if [ -n "$stage" ]; then
        idx="$(page_index "$name" || echo "")"
        if [ -n "$idx" ]; then
            xvfb-run -a drawio -x -f png --scale 2 -p "$idx" \
                -o "$stage/spear_${name}.png" "$stage/spear.drawio" \
                --no-sandbox --disable-gpu >/dev/null 2>&1 || true
            [ -s "$stage/spear_${name}.png" ] && cp "$stage/spear_${name}.png" "$out"
        fi
    fi

    if [ ! -s "$out" ] && [ -f "spear_${name}.svg" ]; then
        if command -v rsvg-convert >/dev/null 2>&1; then
            rsvg-convert -d "$density" -p "$density" -b white \
                -o "$out" "spear_${name}.svg"
        elif command -v convert >/dev/null 2>&1; then
            convert -density "$density" -background white \
                "spear_${name}.svg" "$out"
        fi
    fi

    if [ -s "$out" ]; then
        echo "  spear_${name}.png  ($(stat -c%s "$out") bytes)"
    else
        echo "  FAILED: spear_${name}.png  (no drawio CLI, no rsvg-convert, no convert)" >&2
    fi
done
