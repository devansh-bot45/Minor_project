// Build Minor_Project_Report_COMPLETE.docx from report_part1..4.md in the layout of
// "Minor Project format - final 2026.docx" (IIIT Bhopal): Times New Roman, main headings 16 pt,
// sub-headings 14 pt, text 12 pt, captions 11 pt, references 10 pt, line spacing 1.15.
//
// Usage (from the repository root):
//   NODE_PATH=<folder containing node_modules/docx> node report_build/build_report.js <output.docx> [pages.json]
// REPORT_SRC=<folder> reads report_part1..4.md, diagrams/ and figures/ from that folder instead
// (used for report_final/); lists are then justified like the body text.
// pages.json (optional) maps bookmark ids to page labels; it fills the page column of the
// table of contents and of the lists of figures and tables (the cells are PAGEREF fields, so
// Word can also update them itself).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell, WidthType, ShadingType,
  BorderStyle, ImageRun, Footer, Header, PageNumber, LevelFormat, TableLayoutType, NumberFormat, Bookmark,
  SimpleField, LineRuleType, VerticalAlign, HighlightColor, PageReference, Run, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const [, , outPath, pagesPath] = process.argv;
const PAGES = pagesPath && fs.existsSync(pagesPath) ? JSON.parse(fs.readFileSync(pagesPath, "utf8")) : {};
const SRC = process.env.REPORT_SRC ? path.resolve(process.env.REPORT_SRC) : null;
const PARTS_DIR = SRC || ROOT;
const DIAGRAM_DIR = SRC ? path.join(SRC, "diagrams") : path.join(__dirname, "diagrams");
const FIGURE_BASE = SRC || __dirname;
const LIST_ALIGN = SRC ? AlignmentType.JUSTIFIED : AlignmentType.LEFT;
const md = ["report_part1.md", "report_part2.md", "report_part3.md", "report_part4.md"]
  .map((f) => fs.readFileSync(path.join(PARTS_DIR, f), "utf8")).join("\n");
const lines = md.split("\n");

const FONT = "Times New Roman", MONO = "Courier New";
const PAGE_W = 11906, PAGE_H = 16838, M = 1440;           // A4, 1-inch margins (template body section)
const CW = PAGE_W - 2 * M;                                 // 9026 DXA
const SZ = { text: 24, h1: 32, h2: 28, h3: 24, caption: 22, ref: 20, table: 20, code: 18 };
const LOGO = path.join(__dirname, "assets", "iiitb_logo.jpeg");

// ── inline text: **bold**, *italic*, `code`, [[placeholder]], x_{sub}, x^{sup} ─────
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\[[^\]]+\]\]|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  const push = (t, extra = {}) => { if (t) out.push(new TextRun({ text: t, font: FONT, ...base, ...extra })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) runs(tok.slice(2, -2), { ...base, bold: true }).forEach((r) => out.push(r));
    else if (tok.startsWith("*")) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: MONO, size: (base.size || SZ.text) - 2 });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subScript: true });
    else if (tok.startsWith("^{")) push(tok.slice(2, -1), { superScript: true });
    else push(`[${tok.slice(2, -2)}]`, { highlight: HighlightColor.YELLOW });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

// ── collected structure ─────────────────────────────────────────────────────
const toc = [];        // {bm, kind: front|chapter|section|back, label, title}
const figs = [], tabs = [];
let bmCount = 0;
const bmId = (s) => "bm_" + s.replace(/[^A-Za-z0-9]/g, "_").slice(0, 30) + "_" + (++bmCount);
const pageOf = (bm) => PAGES[bm] || "";

const FRONT = ["CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT", "STATEMENT OF PROJECT ORIGIN AND CONTRIBUTIONS",
  "AREA OF WORK", "TABLE OF CONTENTS", "LIST OF FIGURES", "LIST OF TABLES", "ABSTRACT"];
const BACK = ["REFERENCES", "LIST OF PUBLICATIONS", "PLAGIARISM REPORT"];
const titleCase = (s) => s.toLowerCase().replace(/\b([a-z])/g, (c) => c.toUpperCase()).replace(/\bAnd\b/g, "and").replace(/\bOf\b/g, "of");

// ── block builders ──────────────────────────────────────────────────────────
const para = (text, o = {}) => new Paragraph({
  alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: 276 },
  indent: o.indent, keepNext: o.keepNext, children: runs(text, o.run || {}),
});

function heading(text, level, bm, o = {}) {
  const sizes = { 1: SZ.h1, 2: SZ.h2, 3: SZ.h3 };
  const kids = o.chapter
    ? [new TextRun({ text: o.chapter, bold: true, size: SZ.h1, font: FONT }), new TextRun({ text: text, bold: true, size: SZ.h1, font: FONT, break: 1 })]
    : runs(text, { bold: true, size: sizes[level] });
  return new Paragraph({
    alignment: level === 1 ? AlignmentType.CENTER : AlignmentType.LEFT,
    pageBreakBefore: !!o.pageBreak, keepNext: o.keepNext !== false, keepLines: true,
    spacing: level === 1 ? { before: 0, after: 360, line: 276 } : { before: level === 2 ? 280 : 200, after: 120, line: 276 },
    outlineLevel: level - 1,
    children: bm ? [new Bookmark({ id: bm, children: kids })] : kids,
  });
}

function caption(text, bm, after) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: !after, keepLines: true,
    spacing: { before: after ? 60 : 200, after: after ? 240 : 100 },
    children: [new Bookmark({ id: bm, children: runs(text, { bold: true, size: SZ.caption }) })],
  });
}

function image(file, maxWpx = 602, maxHpx = 800, scaleDiv = 2) {
  const buf = fs.readFileSync(file);
  let w = buf.readUInt32BE(16) / scaleDiv, h = buf.readUInt32BE(20) / scaleDiv;
  const s = Math.min(1, maxWpx / w, maxHpx / h);
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(w * s), height: Math.round(h * s) },
      altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function splitRow(r) { return r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()); }
const plainLen = (s) => s.replace(/\*\*|`|\[\[|\]\]/g, "").length;

function table(rows) {
  let cells = rows.map(splitRow);
  const plain = cells[0][0] === "{plain}";
  const align = cells[1].map((c) => (c.startsWith(":") && c.endsWith(":") ? AlignmentType.CENTER : c.endsWith(":") ? AlignmentType.RIGHT : AlignmentType.LEFT));
  const header = plain ? null : cells[0];
  const body = cells.slice(2);
  const n = cells[0].length;
  const all = header ? [header, ...body] : body;
  const weights = Array.from({ length: n }, (_, i) => {
    const lens = all.map((r) => Math.min(60, plainLen(r[i] || "")));
    return Math.max(4, 0.6 * Math.max(...lens) + 0.4 * lens.reduce((a, b) => a + b, 0) / lens.length);
  });
  const minW = Array.from({ length: n }, (_, i) => {
    const word = Math.max(...all.map((r) => Math.max(0, ...(r[i] || "").replace(/\*\*/g, "").split(/\s+/).map((w) => w.length))));
    return Math.min(16, word) * 118 + 170;
  });
  const sumMin = minW.reduce((a, b) => a + b, 0);
  let widths;
  if (sumMin >= CW) widths = minW.map((w) => Math.floor(w * CW / sumMin));
  else {
    // give every column its minimum, then share the remaining width in proportion to the content
    const extra = weights.map((w, i) => Math.max(0, w / weights.reduce((a, b) => a + b, 0) * CW - minW[i]));
    const tot = extra.reduce((a, b) => a + b, 0) || 1;
    widths = minW.map((m, i) => Math.floor(m + extra[i] / tot * (CW - sumMin)));
  }
  widths[n - 1] += CW - widths.reduce((a, b) => a + b, 0);
  const b = plain ? { top: none, bottom: none, left: none, right: none } : { top: border, bottom: border, left: border, right: border };
  const mk = (r, isHead) => new TableRow({
    tableHeader: isHead, cantSplit: true,
    children: Array.from({ length: n }, (_, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA }, borders: b, verticalAlign: VerticalAlign.TOP,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      shading: isHead ? { type: ShadingType.CLEAR, fill: "D9D9D9", color: "auto" } : undefined,
      children: [new Paragraph({ alignment: plain ? align[i] : AlignmentType.LEFT, spacing: { before: 0, after: 0, line: 252 },
        children: runs(r[i] || "", { size: plain ? SZ.text : SZ.table, bold: isHead || undefined }) })],
    })),
  });
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    borders: plain ? { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none } : undefined,
    rows: [...(header ? [mk(header, true)] : []), ...body.map((r) => mk(r, false))] }),
    new Paragraph({ spacing: { before: 0, after: 160 }, children: [] })];
}

function code(block) {
  const bd = { style: BorderStyle.SINGLE, size: 4, color: "999999", space: 4 };
  return block.map((l, k) => new Paragraph({
    keepLines: true, keepNext: k < block.length - 1,
    spacing: { before: k === 0 ? 120 : 0, after: k === block.length - 1 ? 60 : 0, line: 220, lineRule: LineRuleType.EXACT },
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
    border: { top: bd, bottom: bd, left: bd, right: bd }, indent: { left: 80, right: 80 },
    children: [new TextRun({ text: l || " ", font: MONO, size: SZ.code })],
  }));
}

// ```algorithm block: ruled box with a bold title, bold Input/Output labels and numbered steps;
// leading spaces after "n:" (four per level) indent the step
function algorithm(block) {
  const lines = block.filter((l) => l.trim());
  const thick = { style: BorderStyle.SINGLE, size: 12, color: "000000", space: 2 };
  const thin = { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 2 };
  const base = { size: SZ.caption };
  return lines.map((l, k) => {
    const last = k === lines.length - 1;
    const o = { keepLines: true, keepNext: !last, spacing: { before: k === 0 ? 200 : 0, after: last ? 240 : 20, line: 252 } };
    if (k === 0) {
      const t = l.match(/^(Algorithm \d+:)\s*(.*)$/);
      return new Paragraph({ ...o, spacing: { ...o.spacing, after: 60 }, border: { top: thick, bottom: thin },
        children: [new TextRun({ text: t[1] + " ", bold: true, font: FONT, size: SZ.caption }), ...runs(t[2], base)] });
    }
    const border = last ? { bottom: thick } : undefined;
    const io = l.match(/^(Input|Output):\s*(.*)$/);
    if (io) return new Paragraph({ ...o, border, indent: { left: 900, hanging: 900 }, tabStops: [{ type: TabStopType.LEFT, position: 900 }],
      children: [new TextRun({ text: io[1] + ":", bold: true, font: FONT, size: SZ.caption }), new TextRun({ text: "\t", font: FONT }), ...runs(io[2], base)] });
    const st = l.match(/^(\d+):( *)(.*)$/);
    const level = st ? Math.max(0, Math.floor((st[2].length - 1) / 4)) : 0;
    const left = 480 + level * 400;
    return new Paragraph({ ...o, border, indent: { left, hanging: left }, tabStops: [{ type: TabStopType.LEFT, position: left }],
      children: st ? [new TextRun({ text: st[1] + ":", font: FONT, size: SZ.caption }), new TextRun({ text: "\t", font: FONT }), ...runs(st[3], base)]
        : runs(l, base) });
  });
}

// $$ equation $$ (n.n): centred, with the equation number at the right margin
function equation(eq, no) {
  return new Paragraph({ keepLines: true, spacing: { before: 80, after: 120, line: 276 },
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(CW / 2) }, { type: TabStopType.RIGHT, position: CW - 100 }],
    children: [new TextRun({ text: "\t", font: FONT }), ...runs(eq.trim()), new TextRun({ text: no ? "\t" + no : "", font: FONT })] });
}

function pageRef(bm) {
  // A complex PAGEREF field whose cached result is the page label found in the rendered PDF.
  // Word refreshes it when it updates fields; other applications show the cached label.
  const pr = new PageReference(bm);
  const parts = pr.root.filter((c) => c && c.rootKey !== "w:rPr");           // begin, instruction, separate, end
  const text = new TextRun(pageOf(bm) || " ").root.find((c) => c && c.rootKey === "w:t");
  const run = new Run({ font: FONT, size: SZ.table + 2 });
  run.root.push(parts[0], parts[1], parts[2], text, parts[3]);
  return run;
}

// generated tables: contents, figures, tables (page column = PAGEREF fields with cached labels)
function listTable(headers, rows, widths) {
  const b = { top: border, bottom: border, left: border, right: border };
  const cell = (children, w, shade) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders: b,
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    shading: shade ? { type: ShadingType.CLEAR, fill: "D9D9D9", color: "auto" } : undefined, children });
  const p = (kids, alignC, indent) => new Paragraph({ alignment: alignC ? AlignmentType.CENTER : AlignmentType.LEFT,
    indent: indent ? { left: indent } : undefined, spacing: { before: 0, after: 0, line: 252 }, children: kids });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows: [
    new TableRow({ tableHeader: true, children: headers.map((h, i) => cell([p([new TextRun({ text: h, bold: true, font: FONT, size: SZ.table + 2 })], true)], widths[i], true)) }),
    ...rows.map((r) => new TableRow({ cantSplit: true, children: [
      cell([p([new TextRun({ text: r.no || "", font: FONT, size: SZ.table + 2, bold: r.bold })], true)], widths[0]),
      cell([p([new TextRun({ text: r.title, font: FONT, size: SZ.table + 2, bold: r.bold })], false, r.indent)], widths[1]),
      cell([p([pageRef(r.bm)], true)], widths[2]),
    ] })),
  ] });
}

// ── walk the Markdown ───────────────────────────────────────────────────────
const cover = {}; const coverStudents = [];
const front = [], main = [];
let nextMaxH = null;
let target = null, i = 0, diagram = 0, numInstance = 0, prevNumbered = false, firstInSection = { front: true, main: true };
const placeholders = { toc: null, lof: null, lot: null };

while (i < lines.length) {
  const line = lines[i], t = line.trim();
  if (line.startsWith("# COVER")) { target = "cover"; i++; continue; }
  if (target === "cover" && !line.startsWith("## ")) {
    const m = t.match(/^([A-Z_]+):\s*(.*)$/);
    if (m) { if (m[1] === "STUDENT") coverStudents.push(m[2]); else cover[m[1]] = m[2]; }
    i++; continue;
  }
  if (line.startsWith("## ")) {
    const h = line.slice(3).trim();
    const ch = h.match(/^CHAPTER (\d+):\s*(.*)$/);
    if (FRONT.includes(h)) {
      target = "front";
      const bm = bmId(h);
      if (h !== "TABLE OF CONTENTS") toc.push({ bm, kind: "front", title: titleCase(h) });
      if (h === "CERTIFICATE" || h === "DECLARATION") {
        front.push(new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: !firstInSection.front, spacing: { after: 120 },
          children: [new TextRun({ text: "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", bold: true, size: 32, font: FONT }),
            new TextRun({ text: "BHOPAL (M.P.)", bold: true, size: 32, font: FONT, break: 1 })] }));
        front.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new ImageRun({ type: "jpg",
          data: fs.readFileSync(LOGO), transformation: { width: 110, height: 110 }, altText: { title: "IIIT Bhopal logo", description: "IIIT Bhopal logo", name: "logo" } })] }));
        front.push(heading(h, 1, bm));
      } else front.push(heading(h, 1, bm, { pageBreak: !firstInSection.front,
        keepNext: !["TABLE OF CONTENTS", "LIST OF FIGURES", "LIST OF TABLES"].includes(h) }));
      firstInSection.front = false;
      if (h === "TABLE OF CONTENTS") placeholders.toc = front.length, front.push(null);
      if (h === "LIST OF FIGURES") placeholders.lof = front.length, front.push(null);
      if (h === "LIST OF TABLES") placeholders.lot = front.length, front.push(null);
    } else {
      target = "main";
      const bm = bmId(h);
      if (ch) {
        toc.push({ bm, kind: "chapter", no: ch[1], title: titleCase(ch[2]).replace("Tools and Technology Used", "Tools and Technology Used") });
        main.push(heading(ch[2], 1, bm, { chapter: `CHAPTER ${ch[1]}`, pageBreak: !firstInSection.main }));
      } else {
        toc.push({ bm, kind: "back", title: titleCase(h) });
        main.push(heading(h, 1, bm, { pageBreak: !firstInSection.main }));
      }
      firstInSection.main = false;
    }
    prevNumbered = false; i++; continue;
  }
  const out = target === "front" ? front : main;
  if (!t) { prevNumbered = false; i++; continue; }

  if (line.startsWith("### ")) {
    const h = line.slice(4).trim(); const bm = bmId(h);
    toc.push({ bm, kind: "section", title: h });
    out.push(heading(h, 2, bm)); i++; continue;
  }
  if (line.startsWith("#### ")) { out.push(heading(line.slice(5).trim(), 3)); i++; continue; }

  const hint = t.match(/^<!--\s*maxh=(\d+)\s*-->$/);
  if (hint) { nextMaxH = Number(hint[1]); i++; continue; }
  if (t.startsWith("```")) {
    const lang = t.slice(3).trim(); const block = [];
    for (i++; i < lines.length && !lines[i].trim().startsWith("```"); i++) block.push(lines[i]);
    i++;
    if (lang === "mermaid") {
      out.push(image(path.join(DIAGRAM_DIR, `d${String(++diagram).padStart(2, "0")}.png`), 602, nextMaxH || 800));
      nextMaxH = null;
    }
    else if (lang === "algorithm") out.push(...algorithm(block));
    else out.push(...code(block));
    continue;
  }
  const img = t.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
  if (img) { out.push(image(path.join(FIGURE_BASE, img[1]), 560, 760, 200 / 96)); i++; continue; }
  const eq = t.match(/^\$\$(.+)\$\$\s*(\(\d+\.\d+\))?$/);
  if (eq) { out.push(equation(eq[1], eq[2])); i++; continue; }

  if (t.startsWith("|")) {
    const rows = [];
    for (; i < lines.length && lines[i].trim().startsWith("|"); i++) rows.push(lines[i]);
    out.push(...table(rows)); continue;
  }
  const cap = t.match(/^\*\*((Figure|Table) (\d+\.\d+): (.*))\*\*$/);
  if (cap) {
    const bm = bmId(cap[2] + cap[3]);
    (cap[2] === "Figure" ? figs : tabs).push({ bm, no: cap[3], title: cap[4] });
    out.push(caption(cap[1], bm, cap[2] === "Figure")); i++; continue;
  }
  if (/^- /.test(t)) {
    out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: LIST_ALIGN,
      spacing: { after: 60, line: 276 }, children: runs(t.slice(2)) }));
    i++; continue;
  }
  const num = t.match(/^(\d+)\. (.*)$/);
  if (num) {
    if (!prevNumbered) numInstance++;
    out.push(new Paragraph({ numbering: { reference: "numbers", level: 0, instance: numInstance }, alignment: LIST_ALIGN,
      spacing: { after: 80, line: 276 }, children: runs(num[2]) }));
    prevNumbered = true; i++; continue;
  }
  if (/^\[\d+\]/.test(t)) {
    out.push(new Paragraph({ indent: { left: 567, hanging: 567 }, spacing: { after: 80, line: 252 }, children: runs(t, { size: SZ.ref }) }));
    i++; continue;
  }
  out.push(para(t, { align: /^Date:|^\[\[|^Official dataset page:/.test(t) ? AlignmentType.LEFT : AlignmentType.JUSTIFIED }));
  i++;
}

// fill the generated lists
const tocRows = toc.map((e) => e.kind === "chapter" ? { ...e, bold: true }
  : e.kind === "section" ? { ...e, no: "", indent: 360 } : { ...e, no: "" });
// each list table is followed by an empty paragraph, so that the next heading's page break
// is not attached to the table itself (Pages otherwise moves the table to a new page)
const spacer = () => new Paragraph({ spacing: { before: 0, after: 0 }, children: [] });
front.splice(placeholders.lot, 1, spacer(), listTable(["Table No.", "Description", "Page No."], tabs, [1100, 6826, 1100]), spacer());
front.splice(placeholders.lof, 1, spacer(), listTable(["Fig. No.", "Description", "Page No."], figs, [1100, 6826, 1100]), spacer());
front.splice(placeholders.toc, 1, spacer(), listTable(["S.No.", "Title", "Page No."], tocRows, [900, 7026, 1100]), spacer());

// ── cover page (template sizes) ─────────────────────────────────────────────
const c = (text, size, bold, after = 120, before = 0) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before, after },
  children: runs(text, { size, bold }) });
const coverChildren = [
  c(cover.TITLE, 40, true, 360, 240),
  c("MINOR PROJECT REPORT", 28, true, 120),
  c("Submitted in partial fulfillment for the award of the degree of", 28, true, 120),
  c("BACHELOR OF TECHNOLOGY", 32, true, 60),
  c(cover.DEPARTMENT, 28, true, 240),
  c("Submitted to", 28, false, 60),
  c("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", 32, true, 0),
  c("BHOPAL (M.P.)", 32, true, 160),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new ImageRun({ type: "jpg", data: fs.readFileSync(LOGO),
    transformation: { width: 190, height: 190 }, altText: { title: "IIIT Bhopal logo", description: "IIIT Bhopal logo", name: "logo" } })] }),
  c("Submitted by", 32, true, 60),
  ...coverStudents.map((s) => c(s, 28, false, 40)),
  c("Under the supervision of", 32, true, 60, 200),
  c(cover.SUPERVISOR, 28, false, 0), c(cover.SUPERVISOR_DESIGNATION, 28, false, 0), c(cover.SUPERVISOR_DEPARTMENT, 28, false, 200),
  c(cover.DATE, 28, true, 0),
];

const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] });
const pageMargins = { top: M, right: M, bottom: M, left: M, header: 720, footer: 720 };

const doc = new Document({
  creator: "HAR minor project", title: cover.TITLE, description: "Minor project report",
  features: { updateFields: true },
  styles: { default: { document: { run: { font: FONT, size: SZ.text } } } },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1418, right: 1418, bottom: 1418, left: 1701, header: 720, footer: 720 } } },
      headers: { default: new Header({ children: [] }) }, footers: { default: new Footer({ children: [] }) }, children: coverChildren },
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: pageMargins, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer() }, children: front },
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: pageMargins, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: footer() }, children: main },
  ],
});

fs.writeFileSync(path.join(path.dirname(outPath), path.basename(outPath, ".docx") + ".structure.json"),
  JSON.stringify({ toc, figs, tabs }, null, 1));
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log(JSON.stringify({ out: outPath, bytes: buf.length, diagrams: diagram, figures: figs.length, tables: tabs.length,
    toc_entries: toc.length, page_labels_known: Object.keys(PAGES).length }));
});
