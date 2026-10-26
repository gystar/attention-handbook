#!/usr/bin/env python3
"""Chapter 8 constructed examples; vector sources plus PNG for GitBook."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/kv-cache'
OUT.mkdir(parents=True,exist_ok=True)
def start(w,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
def text(p,x,y,t,size=17):p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{escape(t)}</text>')
def rect(p,x,y,w,h,fill,stroke='#aab7c4'):
 p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}"/>')
def save(p,name):
 p.append('</g></svg>');path=OUT/(name+'.svg');path.write_text('\n'.join(p)+'\n');GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(OUT/(name+'.png')),'png',[],[])
p=start(1000,380)
text(p,215,35,'Logical sequence',20);text(p,740,35,'Physical KV blocks',20)
text(p,215,62,'9 valid positions; block size = 4',15)
for row,(label,positions) in enumerate([(0,[0,1,2,3]),(1,[4,5,6,7]),(2,[8,None,None,None])]):
 y=100+row*75;text(p,40,y+27,str(label),16)
 for col,t in enumerate(positions):
  rect(p,75+col*65,y,62,44,'#e9f1f7' if t is not None else '#f4f4f4');text(p,106+col*65,y+28,str(t) if t is not None else '—')
physical=[2,5,7]
for row,block in enumerate(physical):
 y=100+row*75;text(p,570,y+27,str(block),16)
 positions={7:[0,1,2,3],2:[4,5,6,7],5:[8,None,None,None]}[block]
 for col,t in enumerate(positions):
  rect(p,615+col*65,y,62,44,'#e9f1f7' if t is not None else '#f4f4f4');text(p,646+col*65,y+28,str(t) if t is not None else '—')
for logical,block in enumerate([7,2,5]):
 y1=122+logical*75;y2=122+physical.index(block)*75
 p.append(f'<path d="M 350 {y1} C 440 {y1}, 440 {y2}, 530 {y2}" fill="none" stroke="#547795" stroke-width="2"/>')
text(p,500,340,'Block table: [7, 2, 5]     — allocated, not valid',16)
text(p,500,365,'Position 6 → physical block 2, offset 2 (zero-based)',16)
save(p,'paged-mapping')
p=start(1000,430)
colors=['#d4e4f1','#f4dfc4','#d7e8d9','#e6dcec']
for x,title,axis in [(100,'Per-token groups','row'),(585,'Per-channel groups','column')]:
 text(p,x+150,35,title,20);text(p,x+150,64,'Group size = 4',15)
 for j in range(4):text(p,x+34+j*76,99,f'c{j}',15)
 for i in range(4):
  text(p,x-30,140+i*53,f't{i}',15)
  for j in range(4):
   rect(p,x+j*76,111+i*53,69,46,colors[i if axis=='row' else j]);text(p,x+34+j*76,140+i*53,f'x{i}{j}',15)
 text(p,x+150,352,'One row shares a, Δ' if axis=='row' else 'One column shares a, Δ',17)
text(p,500,397,'Rows: token positions     Columns: channels     Color: quantization group',16)
save(p,'quantization-groups')
