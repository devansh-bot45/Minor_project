# Report build

`Minor_Project_Report_COMPLETE.docx` is generated from `report_part1.md` … `report_part4.md`
in the layout of `Minor Project format - final 2026.docx` (Times New Roman; headings 16/14 pt,
text 12 pt, captions 11 pt, references 10 pt, line spacing 1.15). Edit the Markdown parts, not the DOCX.

| File | Purpose |
|---|---|
| `build_report.js` | Markdown → DOCX (cover, certificate, declaration, contents and lists with page numbers, chapters, references) |
| `render_diagrams.sh` | Renders the ```` ```mermaid ```` blocks to `diagrams/dNN.png` (needs `mmdc` and Google Chrome) |
| `make_figures.py` | Draws Figures 8.1 and 8.2 from `backend/model_v2/comparison.json` |
| `pages.json` | Page labels used for the contents/list tables (taken from the rendered PDF) |
| `assets/iiitb_logo.jpeg` | Institute logo, copied from the format template |

Rebuild (from the repository root):

```bash
backend/.venv/bin/python report_build/make_figures.py
MMDC=/path/to/mmdc sh report_build/render_diagrams.sh
NODE_PATH=/path/to/node_modules node report_build/build_report.js Minor_Project_Report_COMPLETE.docx report_build/pages.json
```

`build_report.js` needs the npm package `docx` (version 9). The page-number cells are PAGEREF fields:
Word refreshes them when it updates fields (it asks when the document is opened). After large edits,
regenerate `pages.json` from a PDF export or simply let Word update the fields.

Placeholders that the authors must fill in before submission are written as `[[...]]` and appear
highlighted in yellow in the DOCX.
