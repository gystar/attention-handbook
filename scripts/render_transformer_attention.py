#!/usr/bin/env python3
"""Render original D2L SVGs on white and draw position-access masks.
Requires PyGObject/GdkPixbuf and Pillow; SVGs remain editable source assets.
"""
from pathlib import Path
from html import escape
import gi
from PIL import Image
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
OUT = Path(__file__).resolve().parents[1] / 'book/assets/transformer-attention'
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="390" viewBox="0 0 1080 390">', '<rect width="1080" height="390" fill="white"/>', '<g font-family="DejaVu Sans, sans-serif" fill="#233647">']
def text(x, y, s, size=16, anchor='middle'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(s)}</text>')
for origin, title, rows, causal, subtitle in [(80, 'Encoder Self-Attention', 4, False, 'Same sequence'), (430, 'Causal Self-Attention', 4, True, 'Same sequence'), (780, 'Cross-Attention', 2, False, 'Query: target  /  Key: source')]:
    text(origin+90, 30, title, 18)
    text(origin+90, 58, subtitle, 13)
    text(origin+90, 88, 'Key position', 14)
    text(origin-38, 129, 'Q', 16)
    for j in range(4):
        text(origin+j*45+22, 111, str(j+1), 14)
    for i in range(rows):
        text(origin-16, 149+i*45, str(i+1), 14)
        for j in range(4):
            allowed = not causal or j<=i
            parts.append(f'<rect x="{origin+j*45}" y="{120+i*45}" width="43" height="43" rx="3" fill="{"#47778e" if allowed else "#eef0f2"}" stroke="#d0d8de"/>')
            if not allowed:
                parts.append(f'<path d="M{origin+j*45+16} {120+i*45+16} l11 11 m0 -11 l-11 11" stroke="#9aa5ad" stroke-width="1.4"/>')
text(540, 337, 'Rows: Query positions     Columns: Key positions', 15)
parts.append('<rect x="350" y="356" width="18" height="18" fill="#47778e"/><rect x="557" y="356" width="18" height="18" fill="#eef0f2" stroke="#d0d8de"/>')
text(380,370,'Allowed',14,'start');text(587,370,'Masked',14,'start')
parts.append('</g></svg>')
(OUT/'attention-masks.svg').write_text('\n'.join(parts)+'\n')
for name in ['d2l-transformer', 'd2l-multi-head-attention', 'attention-masks']:
    source, target = OUT/f'{name}.svg', OUT/f'{name}.png'
    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(source), 1100, -1, True)
    pixbuf.savev(str(target), 'png', [], [])
    with Image.open(target).convert('RGBA') as img:
        white = Image.new('RGBA', img.size, 'white')
        white.alpha_composite(img)
        white.convert('RGB').save(target)
    print(target.relative_to(OUT.parent.parent.parent))
