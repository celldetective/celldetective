# Auditing the docs against the code

Work change by change (phase 1 list), then sweep the whole docs for the
generic checks at the end.

## Per change

1. **Find the code**: the GUI class (`celldetective/gui/...`), the process
   (`celldetective/processes/...`), the library function (`celldetective/*.py`).
   Read the docstrings and the user-facing strings.
2. **Find the pages**: `Grep` `docs/source` for the feature's labels, old and
   new, its option names and its output files. Also the pages that *should*
   mention it: the module overview (`preprocessing.rst`, `segment.rst`,
   `track.rst`, `measure.rst`, `signals-and-events.rst`, `interactions.rst`,
   `table_exploration.rst`, `analysis.rst`), the matching how-to guides, the
   references (`reference/settings.rst`, `reference/menus.rst`,
   `reference/glossary.rst`) and the tutorials (`first-experiment.rst`,
   `adcc-example.rst`).
3. **Compare** every sentence that describes it with the code. Typical drifts:
   - menu paths and action names moved or renamed (`File > Plot...` became
     `Plot > Plot selection...`);
   - shortcuts added, removed or reassigned (grep `setShortcut`, `QKeySequence`);
   - a window replaced by another (config text editor → tabbed editor);
   - button labels ("Export a training sample" vs the real "Export the
     annotation of the current frame");
   - block/section titles in caps (**PREPROCESSING**, not **PROCESSING**);
   - behaviour that depends on a mode or a table type (read all branches);
   - outputs: file names, prefixes, folders (`annotations_<population>` vs
     `labels_<population>`), CSV columns, log files;
   - default values (compare with the widget's `init_value`/`setValue`);
   - steps that were needed and no longer are (e.g. "close and reopen the
     window" after saving, when saving now reloads).
4. **Decide**: fix in place; rewrite the page when the workflow changed; add a
   new guide when the feature is new.

## Code → docs links

```bash
grep -rn "readthedocs.io" celldetective --include=*.py
```

For each URL, check that the page exists in `docs/source` (path without
`.html`) and that an anchor `#x` matches a section title (Sphinx slugifies
titles: "Neighborhood" → `#neighborhood`) or an explicit `.. _x:` label.
Help panels (`open_help(..., docs_url=...)`) and the Help menu use these.
A dead link is fixed in the code, and noted under *Fixed* in the CHANGELOG.

## Figures

```bash
grep -rn "figure::\|image::\|\.gif\|\.png\|\.svg" docs/source --include=*.rst
```

For each figure showing a window that changed, decide: still accurate (keep,
even in an older visual style) / wrong (recapture) / missing (new feature
without a figure → add one when a picture helps). Unreferenced files in
`_static` are not worth recapturing.

## Generic sweep

- Every `:doc:` and `:ref:` resolves (the Sphinx build reports it).
- Every new page is in a toctree.
- `:icon:` names exist in MDI 7.4 (the docs load
  `@mdi/font@7.4.47`); take the name from the code (`MDI6.table_column_plus_after`
  → `table-column-plus-after`).
- Keyboard shortcuts in `reference/menus.rst` match the code.
