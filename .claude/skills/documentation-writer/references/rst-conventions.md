# Writing conventions of the celldetective docs

## Page types (Diátaxis)

- **Tutorials**: `get-started.rst`, `first-experiment.rst`, `adcc-example.rst`.
- **How-to guides**: `how-to-guides/basics/*.rst` and `advanced/*.rst`, each
  listed in the folder's `index.rst` toctree. Title "How to …". Shape:

  ```rst
  How to register stacks
  ======================

  This guide shows you how to <goal>.

  Reference keys: :term:`preprocessing`, :term:`alignment`

  **Prerequisite:** <what must exist first, with a :doc: link>.

  Overview / section
  ------------------

  #. Numbered steps, one action each, UI names in **bold**.
  ```

  Close with `.. note::` / `.. tip::` / `.. seealso::` when useful.
- **Explanation/overview** pages per module (`preprocessing.rst`,
  `segment.rst`, …): what it does and why, then links to the guides.
- **Reference**: `reference/settings.rst` (every option, grouped by module,
  anchored `.. _ref_<x>_settings:`), `reference/menus.rst` (menus and
  shortcuts per window), `reference/glossary.rst` (`:term:` targets).

## Roles and markup

- `:icon:`name,color`` renders an MDI icon, e.g. ``:icon:`image-check,black` ``,
  ``:icon:`plus,#1565c0` ``. Name the button by its icon when it has no text.
- `:blue:`text`` for the text of the blue primary buttons, e.g.
  ``:icon:`plus,#1565c0` :blue:`Add correction` ``.
- `:kbd:`Ctrl+S`` for shortcuts. Menu paths as **Menu > Action...** with the
  exact label, ellipsis included.
- Literals in double backticks: file names, prefixes, column names, values.
- Cross-references: `:doc:` with a path relative to the current page (or
  absolute from `docs/source` with a leading `/`), `:ref:` to labels.

## Figures

- Paths are relative to the page: `_static/...` at the root,
  `../../_static/...` from `how-to-guides/*/`.
- New figures are SVGs in `_static/figures/`, with `:width: 100%`, a
  `:target:` to the same file (click to open full size) and a caption whose
  bold first sentence is the title, without badge numbers. The numbered
  badges are explained by a normal paragraph right after the figure, which
  refers to them inline: "… (1), then … (2)."

## Tone

Plain, precise, second person for instructions. Say what the user sees and
what happens (files written, where), not how the code is organised. Prefer
fixing a wrong sentence over adding a warning next to it. Keep units and
defaults exact, taken from the code.
