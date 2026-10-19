#!/usr/bin/env python3
"""Rasterize the archived D2L SVG on white; retain the original unchanged.

Requires PyGObject and GdkPixbuf with SVG support. Run from any directory.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import gi

gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf

root = Path(__file__).resolve().parents[1]
source = root / 'book/assets/overview/d2l-seq2seq.svg'
target = source.with_suffix('.png')
svg = ET.fromstring(source.read_text())
svg.insert(0, ET.Element('{http://www.w3.org/2000/svg}rect', {
    'width': '100%', 'height': '100%', 'fill': 'white',
}))
with tempfile.TemporaryDirectory() as directory:
    preview = Path(directory) / 'white.svg'
    ET.ElementTree(svg).write(preview)
    GdkPixbuf.Pixbuf.new_from_file_at_scale(str(preview), 1200, -1, True).savev(
        str(target), 'png', [], [])
print(target.relative_to(root))
