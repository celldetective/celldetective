# Documentation figures

The annotated figures of the documentation are SVG files, in
`docs/source/_static/figures/`. Each one embeds its screenshots and keeps its
annotations (arrows, labels, numbered badges) as vector objects, so it can be
opened and touched up in Inkscape, or rebuilt from the scripts here when the
interface changes.

```
capture/          scripts that launch celldetective and capture its windows
screenshots/      the captured windows, as PNG
annotate.py       the house style: shadowed windows, curved arrows, badges, legends
build_figures.py  one function per figure: layout and annotations
```

## Rebuild a figure after a UI change

1. Capture the windows again. The scripts drive the software on a copy of the
   `demo_ricm` demo (*File > Open Demo > Spreading Assay Demo*), on Windows at
   a 100 % display scale:

   ```
   set CELLDETECTIVE_DOCS_EXP=C:\path\to\a\copy\of\demo_ricm
   cd capture
   python capture_registration.py
   ```

   The napari capture runs the `lymphocytes_ricm` model on one frame.

2. Rebuild the SVGs. The documents are built with the
   [CLI-Anything Inkscape harness](https://github.com/HKUDS/CLI-Anything/tree/main/inkscape/agent-harness):

   ```
   pip install "git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=inkscape/agent-harness"
   python build_figures.py              # all figures
   python build_figures.py registration # only the matching ones
   ```

   If Inkscape is installed (set `INKSCAPE` to its executable when it is not on
   the `PATH`), a PNG preview of each figure is also rendered in `preview/`.
   The call uses the Inkscape 0.92 syntax (`-z ... -e`).

If a window moved or grew, update the coordinates of its annotations in
`build_figures.py`: they are given relative to the top-left corner of the
screenshot they point at.
