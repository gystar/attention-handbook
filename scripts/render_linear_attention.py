#!/usr/bin/env python3
"""Native vector teaching figures; illustrative sizes, no performance data."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/linear-attention'
OUT.mkdir(parents=True,exist_ok=True)
def start(w,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
def text(p,x,y,t,size=17):p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{escape(t)}</text>')
def grid(p,x,y,rows,cols,color,label):
 cell=20
 for r in range(rows):
  for c in range(cols):p.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="20" height="20" fill="{color}" stroke="white"/>')
 text(p,x+cols*10,y-13,label)
 text(p,x+cols*10,y+rows*20+24,f'{rows} × {cols}',15)
def save(p,name):
 p.append('</g></svg>');path=OUT/(name+'.svg');path.write_text('\n'.join(p)+'\n');GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(OUT/(name+'.png')),'png',[],[])
p=start(1040,480)
text(p,520,28,'N = 8     feature dimension = 3     value dimension = 2',16)
for y,label in [(95,'Pairwise form'),(315,'State form')]:text(p,90,y+55,label,18)
grid(p,200,70,8,3,'#d7e7ef','ΦQ');text(p,285,150,'×',25);grid(p,320,70,3,8,'#e7dfef','ΦKᵀ');text(p,510,150,'→',25);grid(p,550,70,8,8,'#e6eaf0','Pair scores');text(p,742,150,'×',25);grid(p,780,70,8,2,'#f0dfc5','V');text(p,860,150,'=',25);grid(p,925,70,8,2,'#d9e9dc','Numerator')
grid(p,200,290,3,8,'#e7dfef','ΦKᵀ');text(p,385,370,'×',25);grid(p,420,290,8,2,'#f0dfc5','V');text(p,500,370,'→',25);grid(p,550,290,3,2,'#d9e9dc','State S');text(p,650,370,'then',17);grid(p,720,290,8,3,'#d7e7ef','ΦQ');text(p,810,370,'× S =',20);grid(p,925,290,8,2,'#d9e9dc','Numerator')
save(p,'kernel-reordering')
p=start(1000,390)
text(p,500,30,'k = (1, 0)ᵀ     v = 6     β = 1     α = 0.5',18)
for y,label,vals in [(95,'Additive',[2,4,6,0,8,4]),(195,'Delta Rule',[2,4,4,0,6,4]),(295,'Gated Delta Rule',[1,2,5,0,6,2])]:
 text(p,125,y+32,label,18)
 for col,x in enumerate([300,530,790]):
  for row in range(2):
   value=vals[col*2+row];p.append(f'<rect x="{x}" y="{y+row*30}" width="70" height="30" fill="{("#d7e7ef" if row==0 else "#e9edef")}" stroke="white"/>');text(p,x+35,y+row*30+22,str(value),18)
 text(p,450,y+35,'+',25);text(p,705,y+35,'=',25)
text(p,335,68,'Retained state',16);text(p,565,68,'Write',16);text(p,825,68,'New state',16)
text(p,500,380,'Old state = (2, 4)ᵀ; readout along k is the first row.',16)
save(p,'delta-update')
