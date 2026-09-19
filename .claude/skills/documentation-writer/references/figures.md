# Figures: capture, annotate, embed

Everything lives in `docs/figures` (see its README):

```
capture/harness.py     starts the app like celldetective/__main__.py, grab() helpers
capture/capture_*.py   one script per group of windows → screenshots/*.png
annotate.py            Figure class: shot(), arrow(), box(), badge(), legend(),
                       label(), leader(), render()
build_figures.py       one function per figure; FIGURES list; CLI filter by name
```

## House style

Taken from the older hand-made figures (`maingui.png`,
`tuto_ricm/configure_tracking.png`) and kept consistent since:

- whole windows **with their title bar**, on a transparent background, with a
  soft shadow; several windows may overlap like on a desktop;
- **thick black curved arrows** (`arrow`, width 5) for "this opens that";
- **blue boxes** (`box`, #1565c0, rounded) to frame what is talked about;
- **numbered blue badges** (`badge`) on the UI + a `legend` in free space
  when there are more than ~2 things to point at — clearer than many leader
  lines; thin `leader` lines with a dot for a few isolated labels;
- labels in "DejaVu Sans, Arial, sans-serif", black, 14–17 px (21 px on
  full-screen captures such as napari, which the page scales down ~50 %);
- short lowercase labels ("tune the correlation disk on a frame").

## Workflow for one figure

1. **Capture** (see capture-recipes.md). Choose a state that tells the story:
   options filled, a new row added, a model run, a test computed. Keep the
   window small enough to stay legible once scaled into the ~800 px column
   (≈ 450–650 px wide dialogs; tables up to ~1050 px).
2. **Look at every capture** with `Read` before laying it out: wrong window
   (a blank "python" window means you grabbed the wrong object), occluded by
   another app, scrolled away, status bar overwritten, private paths visible.
3. **Lay out** in `build_figures.py`: `f.shot(file, x, y)` returns the origin;
   write annotation coordinates as `ox + x_in_screenshot`, reading pixel
   positions off the capture. Leave margins for labels and legends.
4. **Build**: `python build_figures.py <name>` → SVG + `preview/<name>.png`.
5. **Look at the preview** and fix: labels over UI text, lines crossing
   labels (label from the right when fanning leaders over a toolbar), badges
   hiding a word, arrows ending inside a window. Rebuild until clean.
6. **Check the size**: SVGs embed the PNG as base64 (+33 %). Figures are
   ~50–400 kB; a full-screen napari capture ~1 MB. Grep `<image` count =
   number of shots (see pitfalls).
7. Embed in the page (SKILL.md snippet) and check it in headless Chrome.

## Pitfalls met before

- The CLI-Anything harness gives the document's default layer the id
  `layer1`, which is also the first id its generator returns: a second layer
  then shares it and every object is written twice. `annotate.py` reuses the
  default layer for screenshots and names the annotation layer explicitly.
- The harness writes image links as `inkscape:href`; `render()` rewrites them
  to `xlink:href` (Inkscape 0.92 and browsers need it).
- Screenshots must stay **embedded**: an SVG shown through `<img>` cannot load
  external files, and Sphinx copies the SVG alone to `_images/`.
- Embed the original PNG bytes; re-encoding with PIL doubled the size.
- Previews render on white (`-b #ffffff -y 1`); the SVG itself is transparent
  — a black preview only means the viewer shows transparency as black.
