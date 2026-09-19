# Documentation figures

The annotated figures of the documentation are SVG files, in
`docs/source/_static/figures/`. Each one embeds its screenshots and keeps its
annotations (arrows, labels, numbered badges) as vector objects, so it can be
opened and touched up in Inkscape, or rebuilt from the scripts here when the
interface changes.

```
capture/          scripts that launch celldetective and capture its windows
screenshots/      the captured windows, as PNG
annotate.py       the house style: shadowed windows, curved arrows, badges, callouts
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

   The fluorescence figures (`capture_measurements.py`, `capture_process.py`)
   run on a copy of the `demo_adcc` demo, set in `CELLDETECTIVE_DOCS_ADCC`.

   On Linux (X11), the windows are rendered by Qt, not grabbed from the screen,
   so a locked or busy screen does not matter; the Ubuntu title bar is drawn
   from `capture/titlebar_ubuntu.png`. Captures that pass `marks=` to `grab()`
   also write the rectangles of the widgets a figure points at to
   `screenshots/<name>.json`, which `build_figures.py` reads with `marks()`.

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

## Style

Point at things with numbered callouts (`Figure.callout`): a tight frame
around the target, its badge centred just outside one side. No legend in the
figure: the page explains the numbers in a paragraph right after it. `config-editor.svg` is the reference.

## Editing an SVG by hand

Touching up a figure in Inkscape is fine, but `build_figures.py` rewrites it
from scratch: port the change to the figure's function (or tell Claude, whose
`/documentation-writer` skill does it) before the next build.
