#!/usr/bin/env python3
"""Constructed hybrid layouts and state snapshots; not model performance plots."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/hybrid-architectures'
OUT.mkdir(parents=True,exist_ok=True)
def start(w,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#637588"/></marker></defs>','<rect width="100%" height="100%" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
def text(p,x,y,s,size=17):p.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{escape(s)}</text>')
def box(p,x,y,w,h,color):p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color}" stroke="#b7c3ce"/>')
def arrow(p,x,y,u,v):p.append(f'<path d="M{x},{y} L{u},{v}" stroke="#637588" stroke-width="1.5" marker-end="url(#arrow)"/>')
def save(p,name):
 p.append('</g></svg>');f=OUT/(name+'.svg');f.write_text('\n'.join(p)+'\n');GdkPixbuf.Pixbuf.new_from_file(str(f)).savev(str(OUT/(name+'.png')),'png',[],[])
p=start(1050,465)
labels={'F':('Full Attention','KV: N positions','#dce8f2'),'W':('Local Attention','KV: min(N, w) positions','#f2e6d5'),'R':('Recurrence','Fixed core + auxiliary state','#dceade')}
for x,title,seq in [(45,'Full Attention',['F','F','F']),(375,'Recurrence + Local',['R','R','W']),(705,'Recurrence + Full',['R','R','F'])]:
 text(p,x+145,30,title,19);text(p,x+145,62,'Input',15)
 for i,kind in enumerate(seq):
  y=90+i*100;arrow(p,x+145,y-20,x+145,y-3);box(p,x,y,290,72,labels[kind][2]);text(p,x+145,y+27,labels[kind][0],18);text(p,x+145,y+53,labels[kind][1],14)
 arrow(p,x+145,362,x+145,389);text(p,x+145,414,'Output',15)
text(p,525,449,'Illustrative layouts; residual connections, normalization and FFNs omitted.',14)
save(p,'layer-state-layouts')
p=start(1080,460)
text(p,190,40,'Shared prefix ends at P',19)
box(p,40,75,300,290,'#f4f6f8')
text(p,190,109,'Snapshot C(P)',19)
items=['Global-layer KV','Local windows + offsets','Recurrent states','Short-convolution history']
for i,label in enumerate(items):
 box(p,65,130+51*i,250,38,['#dce8f2','#f2e6d5','#dceade','#e8e0ef'][i]);text(p,190,155+51*i,label,14)
for y,name in [(90,'Branch A'),(270,'Branch B')]:
 arrow(p,340,220,485,y+45);text(p,565,y-15,name,18)
 box(p,490,y,150,90,'#f4f6f8');text(p,565,y+34,'Restore C(P)',15);text(p,565,y+61,'Private writes',14)
 arrow(p,640,y+45,735,y+45);box(p,740,y,300,90,'#eef3f7');text(p,890,y+34,'Process branch suffix',17);text(p,890,y+63,'Update all layer states',15)
text(p,540,410,'Immutable KV prefixes may be shared; mutable state must be protected.',15)
text(p,540,438,'Snapshots align at the same processed position P across all layers.',15)
save(p,'prefix-state-branches')
