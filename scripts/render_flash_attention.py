#!/usr/bin/env python3
"""Editable chapter 6 teaching example; no measurements or copied artwork."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/flash-attention'
OUT.mkdir(parents=True,exist_ok=True)
p=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="510" viewBox="0 0 1080 510">','<rect width="1080" height="510" fill="white"/>','<g font-family="DejaVu Sans,sans-serif" fill="#243949">']
def text(x,y,s,size=18):
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{escape(s)}</text>')
def rect(x,y,w,h,fill):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#b4c3cc"/>')
text(540,30,'One query · three keys · two-dimensional values',18)
for x,fill,label,lines in [(35,'#edf3fa','Block A',['scores: 0, log 2','values: (1, 0), (0, 1)','m_A = log 2','ℓ_A = 3/2     u_A = (1/2, 1)']), (560,'#eaf4ef','Block B',['score: log 6','value: (1, 1)','m_B = log 6','ℓ_B = 1     u_B = (1, 1)'])]:
    rect(x,55,485,185,fill)
    text(x+242,85,label,20)
    for i,s in enumerate(lines):text(x+242,119+29*i,s)
text(540,278,'Common scale: m = log 6',20)
rect(35,299,485,76,'#edf3fa');rect(560,299,485,76,'#eaf4ef')
text(277,328,'α = exp(log 2 − log 6) = 1/3')
text(277,357,'αℓ_A = 1/2     αu_A = (1/6, 1/3)')
text(802,328,'β = exp(log 6 − log 6) = 1')
text(802,357,'βℓ_B = 1       βu_B = (1, 1)')
text(540,412,'Add:  ℓ = 3/2     u = (7/6, 4/3)',20)
rect(285,435,510,55,'#f3f3f2');text(540,470,'Normalize once:  o = u / ℓ = (7/9, 8/9)',20)
p.append('</g></svg>')
path=OUT/'online-merge.svg';path.write_text('\n'.join(p)+'\n')
GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(OUT/'online-merge.png'),'png',[],[])
