---
name: documentation-writer
description: Bring the celldetective documentation (docs/source, Sphinx) in line with the code, typically before a release. Finds what changed since the last tag, audits the docs for outdated paragraphs, menu paths, button labels, shortcuts, dead links and outdated figures, documents new features, recaptures screenshots by driving the real GUI, annotates them as SVG figures in the house style, updates the CHANGELOG and checks the Sphinx build. Use when asked to update, audit, check or write the docs, refresh screenshots or figures, document a feature, or prepare the docs for a release.
---

# Documentation writer

The docs are Sphinx/reStructuredText in `docs/source`, published on readthedocs.
Figures are self-contained SVGs built from scripted captures of the GUI
(`docs/figures`). Work autonomously through the phases below and report at the
end; ask only when a decision is genuinely the user's.

Never trust the docs, the CHANGELOG or a commit message about what the software
does: **every statement you write or keep must be checked against the code.**
Last time, about a third of the "outdated" findings were not visible in any diff
summary and only came out of reading the code the doc described.

## Phase 1 — Scope: what changed

```bash
git tag --sort=-creatordate | head -5            # last release tag, e.g. v1.5.3b3
git log --oneline <tag>..HEAD --no-merges
git log <tag>..HEAD --no-merges --grep=feat --format="=== %h %s%n%b"
git diff --stat=160 <tag>..HEAD -- . ':!tests'   # gui/*, napari/*, processes/* = user-facing
git diff <tag>..HEAD -- CHANGELOG.md docs/source # what is already documented
```

Build a list of user-facing changes: new windows/panels, renamed or moved
actions, new options, changed behaviour, new outputs (files, columns). The
`[Unreleased]` CHANGELOG section is usually incomplete: treat it as a hint.
Merges of feature branches (e.g. "Merge pull request #37 … feat/…") often hide
a whole feature behind one line — read their files.

## Phase 2 — Audit the docs against the code

Follow `references/audit-checklist.md`. In short:

- For each change, find every page that mentions it (`Grep` in `docs/source`
  for the old label, menu path, shortcut, option name) and every page that
  should.
- Check UI strings against their source: `QAction("…")`, `QPushButton("…")`,
  `QLabel("…")`, `setShortcut`, `addMenu`, `setWindowTitle`, tab names, tooltips.
- Check links the code opens (`docs_url=`, `readthedocs.io` strings in
  `celldetective/`) against existing pages and anchors — a dead one is a code fix.
- Check behaviour claims by reading the code path, including its branches
  (e.g. the Table Explorer plots differently for tables with and without tracks).
- List figures that show a changed window: `grep -rn "figure::\|image::"` and
  look at each one next to the current UI.

## Phase 3 — Write

Follow the conventions of `references/rst-conventions.md` (page types, roles
like `:icon:`/`:blue:`/`:kbd:`, figure directives, relative paths). New
feature → a how-to guide in `how-to-guides/basics|advanced` (add it to that
`index.rst` toctree) + a line in the overview page of its module + its options
in `reference/settings.rst` or `reference/menus.rst` + a glossary touch if a
term is involved. Fix wrong instructions rather than adding notes around them.

## Phase 4 — Figures

Follow `references/figures.md`. The pipeline lives in `docs/figures`
(README there): `capture/*.py` drive the real app and save window screenshots,
`build_figures.py` lays them out and annotates them into
`docs/source/_static/figures/<name>.svg`. For a new figure: write a capture
script (recipes in `references/capture-recipes.md`), add a function to
`build_figures.py`, build, **look at the preview**, iterate on coordinates
until no label overlaps UI text or another line.

Embed as:

```rst
.. figure:: ../../_static/figures/<name>.svg
    :width: 100%
    :target: ../../_static/figures/<name>.svg
    :align: center
    :alt: <what it shows>

    **<Title>.** <caption explaining the numbered badges (1), (2)…>
```

## Phase 5 — Release notes and verification

1. Complete the `[Unreleased]` section of `CHANGELOG.md` (Keep a Changelog:
   Added / Changed / Fixed / Documentation), from the change list of phase 1.
2. Build: `python -m sphinx -b html -q docs/source <scratch>/docbuild`. It must
   be free of new warnings (the `nbsphinx_custom_formats` pickling warning is
   pre-existing). Unresolved `:doc:`/`:ref:` show up here.
3. Render the new pages in a real browser to see the SVGs as readers will:
   `"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new
   --window-size=1200,2600 --screenshot=<out.png> file:///<docbuild>/<page>.html`
   and `Read` the PNG.
4. Do not commit unless asked. Report: what was added/fixed (group by kind),
   code fixes made on the way, figures (re)built, and what was deliberately
   left (e.g. older screenshots still accurate but in the previous style) as
   one offer, not a list of notes.

## Environment facts (Windows machine of the maintainer)

- Python: `C:\ProgramData\anaconda3\python.exe` has celldetective (editable),
  napari, cellpose. Qt runs at 1600×900, 100 % scale.
- Inkscape 0.92 at `C:\Program Files\Inkscape\inkscape.exe` (CLI: `-z in.svg
  -e out.png`; 1.x syntax `--export-filename` fails).
- CLI-Anything Inkscape harness: install in a scratch venv
  `pip install "git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=inkscape/agent-harness"`
  and put its `Lib/site-packages` on `PYTHONPATH` when running `build_figures.py`.
- Demo data: `C:\Users\remy1\Documents\Experiments\demo_ricm` (385 MB) and
  `demo_adcc` (4.9 GB). **Copy** demo_ricm to the scratchpad and point
  `CELLDETECTIVE_DOCS_EXP` to the copy; never capture on the originals.
- Python heredocs editing `.rst`/`.py`: always `open(p, encoding="utf-8")` —
  the default cp1252 codec breaks on `·`, `α`, `δ`, `µ`.
