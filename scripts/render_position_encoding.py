#!/usr/bin/env python3
"""Editable SVG teaching diagrams computed from Chapter 3 equations.
Requires PyGObject/GdkPixbuf. No model scores or experimental measurements.
"""
import math
from html import escape
from pathlib import Path
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
OUT=Path(__file__).resolve().parents[1]/'book/assets/position-encoding'
BLUE='#326a89';GREEN='#34836e';GRAY='#71808b';ORANGE='#b77736'

def start(w,h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<rect width="{w}" height="{h}" fill="white"/>','<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#71808b"/></marker></defs>','<g font-family="Noto Sans CJK SC,DejaVu Sans,sans-serif" fill="#253747">']
def txt(p,x,y,s,size=18,anchor='start',color='#253747'):
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>')
def rect(p,x,y,w,h,fill='#eef3f6'):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="#cbd6df"/>')
def line(p,x,y,x2,y2,color=GRAY,arrow=False,dash=''):
    p.append(f'<path d="M{x} {y}L{x2} {y2}" stroke="{color}" stroke-width="2" fill="none"'+(' marker-end="url(#arrow)"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def circle(p,x,y,r,fill='none',stroke=GRAY):
    p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
def save(name,p):
    p.append('</g></svg>');path=OUT/(name+'.svg');path.write_text('\n'.join(p)+'\n')
    GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(path.with_suffix('.png')),'png',[],[])

p=start(1050,390)
for y,label,items in [(55,'Absolute', ['Embedding + Position','Q / K / V projection','Attention']), (170,'RoPE',['Q / K projection','Rotate Q / K','Scores → Softmax']), (285,'Relative Bias',['Content scores','Add position bias','Softmax → AV'])]:
    txt(p,25,y+35,label,20)
    for x,t in zip([190,485,780],items):
        rect(p,x,y,245,65,'#e9f2ef' if (x==190 and label=='Absolute') or (x==485 and label!='Absolute') else '#f5f7f9')
        txt(p,x+122,y+39,t,17,'middle')
        if x<780:line(p,x+248,y+32,x+287,y+32,arrow=True)
txt(p,25,375,'位置项分别作用于输入、Q/K 或匹配分数；图中省略残差、FFN 与其他网络组件。',16)
save('position-paths',p)

p=start(1000,400)
for cx,positions,title in [(240,(1,3),'位置 1 与 3'),(740,(4,6),'共同平移 3 个位置')]:
    cy=215;r=105;circle(p,cx,cy,r,stroke='#ced8df');line(p,cx-130,cy,cx+135,cy,color='#ced8df');line(p,cx,cy+125,cx,cy-130,color='#ced8df');txt(p,cx,35,title,21,'middle')
    for pos,c,name in zip(positions,[BLUE,GREEN],['q','k']):
        angle=pos*math.pi/6;x=cx+r*math.cos(angle);y=cy-r*math.sin(angle);line(p,cx,cy,x,y,color=c);circle(p,x,y,4,c,c);txt(p,x+(12 if x>=cx else -12),y-10,f'{name}: {pos}π/6',17,'start' if x>=cx else 'end',c)
    txt(p,cx,365,'夹角 π/3，点积 1/2',18,'middle')
txt(p,500,66,'固定 q = k = (1, 0)ᵀ，θ = π/6',17,'middle')
save('rope-relative-rotation',p)

p=start(1000,300);txt(p,30,35,'Query 位置 i = 4；仅展示偏置 −m(i−j)',20)
for col in range(5):txt(p,310+col*135,80,f'j = {col}',17,'middle')
for y,m in [(110,.25),(200,.5)]:
    txt(p,30,y+32,f'Head slope m = {m}',18)
    for j in range(5):
        value=0 if j==4 else -m*(4-j);rect(p,260+j*135,y,100,50,'#e5eef3' if value else '#ecf4ee');txt(p,310+j*135,y+32,f'{value:g}',20,'middle')
save('alibi-bias',p)

p=start(1080,390);x0=280;scale=90
for y,label,points in [(90,'原位置 L = 4',range(4)),(200,'直接外推 L′ = 8',range(8)),(310,'PI，ρ = 2',[i/2 for i in range(8)])]:
    txt(p,20,y+6,label,20);rect(p,x0-10,y-36,4*scale+10,70,'#edf4ef');line(p,x0-10,y,x0+7.6*scale,y,arrow=True)
    for i,value in enumerate(points):
        x=x0+value*scale;circle(p,x,y,5,BLUE,BLUE);txt(p,x,y+27,f'{value:g}',15,'middle')
    line(p,x0+4*scale,y-36,x0+4*scale,y+34,color=ORANGE,dash='4 3')
txt(p,280,370,'绿色区域表示原索引范围 [0, 4)；圆点表示输入 RoPE 的坐标。',17)
save('position-interpolation',p)

p=start(1060,470);x0,y0,w,h=100,365,860,265;d,L,rho,b=64,2048,4,10000
for ratio in [.25,.5,.75,1]:
    y=y0-(ratio-.2)/.85*h;line(p,x0,y,x0+w,y,color='#e2e7eb');txt(p,x0-18,y+6,f'{ratio:g}',16,'end')
line(p,x0,y0,x0+w,y0);line(p,x0,y0,x0,y0-h-10)
txt(p,30,45,'频率比 θ′ᵣ / θᵣ',19);txt(p,540,438,'维度对索引 r（左侧高频，右侧低频）',18,'middle')
for r in [0,8,16,24,31]:
    x=x0+r/31*w;txt(p,x,y0+25,str(r),16,'middle')
for label,color,dash,fn in [('PI',BLUE,'7 4',lambda r:.25),('NTK-aware',ORANGE,'',lambda r:rho**(-2*r/(d-2))),('分频插值',GREEN,'',lambda r:.25+.75*max(0,min(1,(L*b**(-2*r/d)/(2*math.pi)-1)/31)))]:
    pts=[]
    for r in range(32):
        ratio=fn(r);pts.append(f'{x0+r/31*w},{y0-(ratio-.2)/.85*h}')
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
for x,label,c in [(180,'PI',BLUE),(420,'NTK-aware',ORANGE),(730,'分频插值',GREEN)]:
    line(p,x,70,x+40,70,color=c);txt(p,x+50,76,label,18)
txt(p,540,465,'公式示例：d = 64，L = 2048，ρ = 4，b = 10000；分频阈值 α = 1，β = 32。',16,'middle')
save('frequency-scaling',p)
print('Rendered 5 SVG/PNG diagram pairs')
