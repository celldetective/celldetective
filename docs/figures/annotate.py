"""
Annotated documentation figures, built with the CLI-Anything Inkscape harness.

The harness holds the document (a JSON project of layers and objects) and writes
it out as SVG. The SVG is what the docs embed: the screenshots are inlined in it,
the annotations stay vector objects that can be edited in Inkscape. Inkscape
itself only renders a PNG preview to look at. The house style follows
the existing figures of the docs: screenshots of whole windows, laid out on a
transparent background, joined by thick black curved arrows, with short plain
sans-serif labels. The celldetective blue marks what a label points at.
"""

import base64
import io
import math
import os
import re
import subprocess
import sys

from PIL import Image

try:
    from cli_anything.inkscape.core import document as doc_mod
except ImportError:
    sys.exit(
        "The CLI-Anything Inkscape harness is needed: pip install "
        '"git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=inkscape/agent-harness"'
    )
from cli_anything.inkscape.core import shapes, text as text_mod
from cli_anything.inkscape.core import export as export_mod
from cli_anything.inkscape.core.layers import add_layer
from cli_anything.inkscape.utils.svg_utils import generate_id

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "screenshots")
OUT = os.path.join(HERE, "..", "source", "_static", "figures")
PREVIEW = os.path.join(HERE, "preview")
INKSCAPE = os.environ.get("INKSCAPE", "inkscape")

BLUE = "#1565c0"
INK = "#000000"
FONT = "DejaVu Sans, Arial, sans-serif"
ARROW_W = 5


class Figure:
    def __init__(self, name, width, height):
        self.name = name
        self.project = doc_mod.create_document(
            name=name, width=width, height=height, background="none"
        )
        # The document comes with a layer whose id, "layer1", is also the first id
        # the harness generates: take it for the screenshots and name the other one,
        # or the two layers would share an id and every screenshot be written twice.
        shots = self.project["layers"][0]
        shots["name"] = "screenshots"
        self.shots = shots["id"]
        marks = add_layer(self.project, name="annotations")
        marks["id"] = "annotations"
        self.marks = marks["id"]

    # -- content ---------------------------------------------------------
    def shot(self, filename, x, y, crop=None):
        """
        Place a screenshot, embedded, with a soft shadow like a window on a desktop.

        crop : (left, top, right, bottom) in pixels of the screenshot, optional.
        """
        img = Image.open(os.path.join(RAW, filename))
        if crop:
            img = img.crop(crop)
        w, h = img.size
        shapes.add_rect(
            self.project, x=x + 3, y=y + 4, width=w, height=h, rx=6, ry=6,
            style="fill:#000000;fill-opacity:0.18;stroke:none", layer=self.shots,
        )
        if crop:
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            raw = buf.getvalue()
        else:
            # The capture as it is: re-encoding it only makes it bigger.
            with open(os.path.join(RAW, filename), "rb") as fh:
                raw = fh.read()
        data = base64.b64encode(raw).decode("ascii")
        obj = {
            "id": generate_id("image"),
            "name": filename,
            "type": "image",
            "x": x, "y": y, "width": w, "height": h,
            "href": "data:image/png;base64," + data,
            "style": "",
            "layer": self.shots,
        }
        shapes._add_object(self.project, obj)
        shapes.add_rect(
            self.project, x=x, y=y, width=w, height=h,
            style="fill:none;stroke:#8a8a8a;stroke-width:1", layer=self.shots,
        )
        return (x, y, w, h)

    def box(self, x, y, w, h, color=BLUE, width=3, dash=False):
        """Outline the region a label is about."""
        style = f"fill:none;stroke:{color};stroke-width:{width};stroke-linejoin:round"
        if dash:
            style += ";stroke-dasharray:8,5"
        shapes.add_rect(self.project, x=x, y=y, width=w, height=h, rx=6, ry=6,
                        style=style, layer=self.marks)

    def label(self, s, x, y, size=17, anchor="start", weight="normal", color=INK):
        """A label; several lines are separated by newlines."""
        text_mod.add_text(
            self.project, text=s, x=x, y=y, font_family=FONT, font_size=size,
            font_weight=weight, fill=color, text_anchor=anchor, line_height=1.25,
            layer=self.marks,
        )

    def badge(self, n, cx, cy, r=14):
        """A numbered point, blue disk and white figure."""
        shapes.add_circle(self.project, cx=cx, cy=cy, r=r,
                          style=f"fill:{BLUE};stroke:#ffffff;stroke-width:2.5",
                          layer=self.marks)
        text_mod.add_text(self.project, text=str(n), x=cx, y=cy + r * 0.46,
                          font_family=FONT, font_size=r * 1.28, font_weight="bold",
                          fill="#ffffff", text_anchor="middle", layer=self.marks)

    def arrow(self, x0, y0, x1, y1, bend=0.25, color=INK, width=ARROW_W, head=16):
        """A curved arrow from (x0, y0) to (x1, y1); bend > 0 curves to the left."""
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        cx, cy = mx - dy * bend, my + dx * bend
        # Direction at the tip of a quadratic Bezier: from the control point to the end.
        tx, ty = x1 - cx, y1 - cy
        n = math.hypot(tx, ty) or 1
        ux, uy = tx / n, ty / n
        # Stop the line under the head so the tip stays sharp.
        bx, by = x1 - ux * head * 0.8, y1 - uy * head * 0.8
        shapes.add_path(
            self.project, d=f"M {x0:.1f},{y0:.1f} Q {cx:.1f},{cy:.1f} {bx:.1f},{by:.1f}",
            style=f"fill:none;stroke:{color};stroke-width:{width};stroke-linecap:round",
            layer=self.marks,
        )
        px, py = -uy, ux
        hw = head * 0.6
        pts = [
            (x1, y1),
            (x1 - ux * head + px * hw, y1 - uy * head + py * hw),
            (x1 - ux * head - px * hw, y1 - uy * head - py * hw),
        ]
        shapes.add_path(
            self.project,
            d="M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + " Z",
            style=f"fill:{color};stroke:{color};stroke-width:1;stroke-linejoin:round",
            layer=self.marks,
        )

    def legend(self, items, x, y, step=30, size=16, r=12):
        """Numbered lines explaining the badges, one per item."""
        for n, s in items:
            self.badge(n, x + r, y - size * 0.35, r=r)
            self.label(s, x + 2 * r + 10, y, size=size)
            y += step

    def leader(self, x0, y0, x1, y1, color=INK, width=1.6):
        """A thin straight line with a dot, from a label to what it names."""
        shapes.add_line(self.project, x1=x0, y1=y0, x2=x1, y2=y1,
                        style=f"stroke:{color};stroke-width:{width}", layer=self.marks)
        shapes.add_circle(self.project, cx=x1, cy=y1, r=3.5,
                          style=f"fill:{color};stroke:none", layer=self.marks)

    # -- output ----------------------------------------------------------
    def render(self, preview=True):
        """Write the SVG the docs embed and, if Inkscape is there, a PNG preview."""
        os.makedirs(OUT, exist_ok=True)
        svg = os.path.abspath(os.path.join(OUT, self.name + ".svg"))
        export_mod.export_svg(self.project, svg, overwrite=True)
        # The harness writes the image link as inkscape:href; Inkscape 0.92 reads xlink:href.
        with open(svg, encoding="utf-8") as f:
            s = f.read()
        if "xmlns:xlink" not in s:
            s = s.replace("<svg ", '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ', 1)
        s = re.sub(r'\s(?:inkscape|ns\d+):href="', ' xlink:href="', s)
        s = re.sub(r'(xlink:href="[^"]*")\s+href="[^"]*"', r"\1", s)
        with open(svg, "w", encoding="utf-8") as f:
            f.write(s)
        print("wrote", os.path.relpath(svg), f"{os.path.getsize(svg) / 1e3:.0f} kB")
        if preview:
            os.makedirs(PREVIEW, exist_ok=True)
            png = os.path.abspath(os.path.join(PREVIEW, self.name + ".png"))
            try:
                # Inkscape 0.92 syntax; with 1.x use --export-filename instead of -e.
                subprocess.run(
                    [INKSCAPE, "-z", svg, "-e", png, "-b", "#ffffff", "-y", "1"],
                    check=True, capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as e:
                print("no preview:", e)
        return svg
