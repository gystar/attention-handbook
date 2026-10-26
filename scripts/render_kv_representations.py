#!/usr/bin/env python3
"""Chapter 4 editable teaching diagrams, based on the displayed equations.
Requires PyGObject/GdkPixbuf. These are computational schematics, not benchmarks.
"""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/kv-representations'
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

p=start(1100,420)
text(p,550,35,'One head · content branch · identical parameters and attention weights',18)
text(p,550,75,'Expanded K/V',18)
box(p,25,100,200,80,'Cached latents','C: N × d_c','#eaf4ef')
box(p,300,100,200,80,'Expand K and V','K, V: N × d_h')
box(p,575,100,200,80,'Attention','q: 1 × d_h')
box(p,850,100,220,80,'Head output','o: 1 × d_h')
for a,b in [(225,300),(500,575),(775,850)]:arrow(p,a,140,b,140)
text(p,550,235,'Latent computation',18)
box(p,25,260,200,80,'Cached latents','C: N × d_c','#eaf4ef')
box(p,300,260,200,80,'Transform query','q_hat = q U_K')
box(p,575,260,200,80,'Weighted latents','z = softmax(scores) C')
box(p,850,260,220,80,'Project once','o = z U_Vᵀ')
# C enters the score and weighted sum; query transform is a separate input.
arrow(p,500,300,575,300);arrow(p,775,300,850,300)
p.append('<path d="M125 340V365H675V340" stroke="#607789" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
text(p,550,398,'Row-vector diagram; scale and mask are unchanged. C stores each position separately.',16)
save('mla-computation',p)

p=start(1100,470)
text(p,550,35,'One query head · one query position t · N cached positions',18)
box(p,30,80,250,85,'Content cache C','N × d_c','#eaf4ef')
box(p,30,240,250,85,'Position cache K_R','N × d_r','#f5eee5')
box(p,360,80,240,85,'Content scores','q_hat Cᵀ : 1 × N')
box(p,360,240,240,85,'Position scores','q_R K_Rᵀ : 1 × N')
arrow(p,280,122,360,122);arrow(p,280,282,360,282)
box(p,710,160,320,85,'Add → scale → mask → softmax','weights: 1 × N')
p.append('<path d="M600 122H650V202 M600 282H650V202H710" stroke="#607789" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
box(p,710,330,320,85,'Weighted content latents','z = weights C : 1 × d_c')
arrow(p,870,245,870,330)
p.append('<path d="M155 165V205H315V373H710" stroke="#34836e" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
text(p,550,451,'Cached elements per position: d_c + d_r. Softmax is applied once after adding both scores.',16)
save('mla-cache-branches',p)
