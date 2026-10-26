#!/usr/bin/env python3
"""Chapter 5 editable teaching diagrams, based on the displayed equations.
Requires PyGObject/GdkPixbuf. These are computational schematics, not benchmarks.
"""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/gated-attention'
OUT.mkdir(parents=True,exist_ok=True)

def start(w,h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<rect width="{w}" height="{h}" fill="white"/>','<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#607789"/></marker></defs>','<g font-family="DejaVu Sans,Noto Sans CJK SC,sans-serif" fill="#243949">']
def text(p,x,y,s,size=18):
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{escape(s)}</text>')
def box(p,x,y,w,h,title,sub,fill='#edf3f7'):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="#aebfca"/>')
    text(p,x+w/2,y+29,title);text(p,x+w/2,y+55,sub,15)
def arrow(p,x,y,x2,y2):
    p.append(f'<path d="M{x} {y}L{x2} {y2}" stroke="#607789" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
def save(name,p):
    p.append('</g></svg>');path=OUT/(name+'.svg');path.write_text('\n'.join(p)+'\n')
    GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(OUT/(name+'.png')),'png',[],[])

p=start(1100,350)
box(p,25,130,150,80,'Input X','N × d')
box(p,265,130,210,80,'Q/K/V → SDPA','H: N × (h_q d_v)')
box(p,570,130,210,80,'Multiply','H ⊙ G')
box(p,870,130,205,80,'Output projection','Y = (H ⊙ G) W_O')
arrow(p,175,170,265,170);arrow(p,475,170,570,170);arrow(p,780,170,870,170)
box(p,570,20,210,80,'Gate projection','G = sigmoid(X W_G)','#eaf4ef')
p.append('<path d="M100 130V60H570" stroke="#34836e" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
arrow(p,675,100,675,130)
text(p,550,280,'The same normalized input supplies the attention branch and the gate.',17)
text(p,550,312,'The output gate acts after SDPA; the already computed attention weights are unchanged.',17)
save('output-gate',p)
p=start(1000,340)
text(p,500,35,'Constructed example · one head · three features',18)
for x,name,vals in [(180,'Head output H',['2','−1','4']),(500,'Gate G',['0.10','0.80','0.25']),(820,'Gated output H ⊙ G',['0.2','−0.8','1.0'])]:
    text(p,x,83,name,18)
    for i,v in enumerate(vals):
        box(p,x-100,105+65*i,200,55,v,'','#eaf4ef' if x==500 else '#edf3f7')
text(p,340,205,'×',28);text(p,660,205,'=',28)
save('gate-example',p)
