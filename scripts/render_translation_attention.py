#!/usr/bin/env python3
"""Draw a pedagogical English–Chinese attention example; weights are illustrative."""
from pathlib import Path
from html import escape
import gi

gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf

root = Path(__file__).resolve().parents[1]
svg = root / 'book/assets/overview/translation-attention-en-zh.svg'
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="420" viewBox="0 0 1120 420">',
'<rect width="1120" height="420" fill="white"/>',
'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7785"/></marker></defs>',
'<g font-family="Noto Sans CJK SC, sans-serif" fill="#243342">']
def text(x,y,label,size=19,color='#243342',anchor='middle'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(label)}</text>')
def box(x,y,w,h,fill,stroke='#d8e1e9'):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}"/>')
def arrow(x1,y1,x2,y2):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#6b7785" stroke-width="1.5" marker-end="url(#arrow)"/>')
text(35,35,'原文：I like tea',21,anchor='start')
text(1085,35,'译文：我喜欢茶',21,anchor='end')
text(145,91,'已生成的译文',18)
text(480,77,'Encoder 各位置状态的读取权重',18)
for x,word in zip([370,480,590],['I','like','tea']):
    text(x,113,word,21)
text(785,91,'加权汇总',18)
text(1000,91,'下一译文单元',18)
for y,prefix,weights,target in [(145,'我',[.1,.8,.1],'喜欢'),(255,'我 喜欢',[.05,.1,.85],'茶')]:
    assert abs(sum(weights)-1)<1e-10
    box(45,y,200,64,'#f4f6f8');text(145,y+40,prefix,23)
    arrow(255,y+32,308,y+32)
    for x,v in zip([325,435,545],weights):
        dark=v>=.5
        box(x,y,90,64,'#286681' if dark else '#edf3f6')
        text(x+45,y+40,f'{v:.2f}',23,'white' if dark else '#425969')
    arrow(650,y+32,709,y+32)
    box(720,y,130,64,'#f1f5f7');text(785,y+40,'Context',20)
    arrow(860,y+32,925,y+32)
    box(940,y,120,64,'#e6f0eb','#8eaf9f');text(1000,y+40,target,24)
    text(900,y-9,'预测',14,'#617080')
text(35,374,'深色表示较高权重；每行权重之和为 1。',16,'#617080','start')
text(1085,400,'教学设定：词语切分与权重用于示意',15,'#617080','end')
parts+=['</g></svg>'];svg.write_text('\n'.join(parts)+'\n')
GdkPixbuf.Pixbuf.new_from_file(str(svg)).savev(str(svg.with_suffix('.png')),'png',[],[])
print(svg.relative_to(root))
