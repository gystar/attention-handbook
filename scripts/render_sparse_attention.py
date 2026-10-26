#!/usr/bin/env python3
"""Chapter 7 connection masks; constructed diagrams, not measured attention weights."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
out=Path(__file__).resolve().parents[1]/'book/assets/sparse-attention'
out.mkdir(parents=True,exist_ok=True)
p=['<svg xmlns="http://www.w3.org/2000/svg" width="1110" height="430" viewBox="0 0 1110 430">','<rect width="1110" height="430" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
def text(x,y,t,size=17):p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{escape(t)}</text>')
for x,title,subtitle,fn in [(55,'Full causal','j ≤ i',lambda i,j:j<=i),(420,'Local causal','width = 3, including self',lambda i,j:0<=i-j<3),(785,'Local + global','bidirectional; global = 1',lambda i,j:abs(i-j)<=1 or i==0 or j==0)]:
 text(x+132,30,title,19);text(x+132,56,subtitle,15)
 for i in range(8):
  text(x-17,108+i*33,str(i+1),13)
  text(x+16+i*33,80,str(i+1),13)
  for j in range(8):
   color='#456f91' if fn(i,j) else '#f1f3f5'
   p.append(f'<rect x="{x+j*33}" y="{87+i*33}" width="32" height="32" fill="{color}"/>')
 text(x+132,380,'Key position →',15)
text(555,417,'Rows: query position     Blue: allowed connection     Gray: masked',16)
p.append('</g></svg>');path=out/'connectivity.svg';path.write_text('\n'.join(p)+'\n')
GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(out/'connectivity.png'),'png',[],[])

p=['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="350" viewBox="0 0 1040 350">','<rect width="1040" height="350" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
text(520,30,'Constructed example: q = (1, 1)',20)
for x,title,keys,bound,actual,fill in [(30,'Page A','(1, 0)   (0, 1)','2','1','#edf3fa'),(540,'Page B','(0.75, 0.75)   (0.75, 0.75)','1.5','1.5','#eaf4ef')]:
 p.append(f'<rect x="{x}" y="55" width="470" height="225" rx="6" fill="{fill}" stroke="#b4c3cc"/>')
 text(x+235,89,title,20);text(x+235,130,'Keys: '+keys,17)
 text(x+235,181,'Page upper bound: '+bound,19)
 text(x+235,226,'Largest actual dot product: '+actual,19)
text(520,319,'Upper-bound ranking: A > B     Actual maximum: B > A',19)
p.append('</g></svg>');path=out/'page-bound.svg';path.write_text('\n'.join(p)+'\n')
GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(out/'page-bound.png'),'png',[],[])
