#!/bin/sh
# Render the ```mermaid blocks of the report parts (in order) to <OUT>/dNN.png.
# Needs @mermaid-js/mermaid-cli (mmdc) and Google Chrome.
# Usage (from the repository root): MMDC=/path/to/mmdc sh report_build/render_diagrams.sh
#   SRC=report_final STYLE=final  reads the parts from report_final/, writes report_final/diagrams/
#   and uses the bold black-and-white style of mermaid_final.json (STYLE=X reads mermaid_X.json)
set -e
SRC=${SRC:-.}
OUT=${OUT:-$( [ "$SRC" = "." ] && echo report_build/diagrams || echo "$SRC/diagrams" )}
mkdir -p "$OUT"
rm -f "$OUT"/d*.mmd "$OUT"/d*.png
cat "$SRC/report_part1.md" "$SRC/report_part2.md" "$SRC/report_part3.md" "$SRC/report_part4.md" | awk -v out="$OUT" '
  /^```mermaid/ { n++; f = sprintf("%s/d%02d.mmd", out, n); inb = 1; next }
  /^```/ && inb  { inb = 0; close(f); next }
  inb            { print > f }'
if [ -n "$STYLE" ]; then
  cp "report_build/mermaid_$STYLE.json" "$OUT/config.json"
else
  printf '{"theme":"default","flowchart":{"htmlLabels":true,"useMaxWidth":false,"nodeSpacing":30,"rankSpacing":32},"sequence":{"useMaxWidth":false,"mirrorActors":false,"actorMargin":20,"width":110,"boxMargin":4,"messageMargin":28},"themeVariables":{"fontFamily":"Times New Roman","fontSize":"16px"}}' > "$OUT/config.json"
fi
printf '{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","args":["--no-sandbox"]}' > "$OUT/puppeteer.json"
for f in "$OUT"/d*.mmd; do
  "${MMDC:-mmdc}" -q -i "$f" -o "${f%.mmd}.png" -s 2 -b white -c "$OUT/config.json" -p "$OUT/puppeteer.json"
done
ls "$OUT"/d*.png | wc -l
