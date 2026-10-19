#!/usr/bin/env python3
"""Deterministic teaching diagrams; no measured performance is depicted."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
out=Path(__file__).resolve().parents[1]/'book/assets/overview'
def start(h):
 global p
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{h}" viewBox="0 0 1080 {h}"><rect width="1080" height="{h}" fill="white"/><g font-family="Noto Sans CJK SC, sans-serif" fill="#243342">']
def text(x,y,t,size=20):p.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(t)}</text>')
def box(x,y,w,h,fill='#edf2f5') :p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#cad5dd"/>')
def save(name):
 p.append('</g></svg>'); path=out/(name+'.svg');path.write_text('\n'.join(p)+'\n');GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(out/(name+'.png')),'png',[],[])
start(330)
text(28,35,'已有序列',19);text(590,35,'模型输出',19);text(875,35,'选取并追加',19)
for y,t,target in [(65,'The cat sat','on'),(145,'The cat sat on','the')]:
 box(28,y,355,58);text(45,y+37,t,24)
 text(401,y+36,'→',27);box(445,y,96,58);text(459,y+36,'LLM',23)
 text(552,y+36,'→',27);box(590,y,240,58);text(610,y+36,'词表上的概率分布',21)
 text(844,y+36,'→',27);box(885,y,160,58,'#e3efe9');text(912,y+37,target,24)
text(28,258,'每轮选出的 Token 成为下一轮输入的一部分。',20)
text(28,302,'教学示例：按 on、the 分步展示；实际 Token 边界取决于 Tokenizer。',17)
save('autoregressive-steps')
start(420)
text(28,32,'完整输入：因果位置对',22);text(578,32,'单步 Decode：读取与缓存',22)
for x,n in [(35,4),(260,8)]:
 cell=22;text(x,72,f'N = {n}',20)
 for i in range(n):
  for j in range(n):box(x+j*cell,92+i*cell,cell-2,cell-2,'#587e98' if j<=i else '#f3f5f7')
 text(x,303,f'{n*(n+1)//2} 个有效位置对',19)
for y,n in [(100,4),(215,8)]:
 text(578,y-15,f'N = {n}',20);text(578,y+23,'q →',22)
 for j in range(n):box(655+j*45,y,40,45,'#d9e8ef');text(665+j*45,y+29,str(j+1),18)
 text(655,y+72,f'读取 {n} 组 KV · 缓存 {n} 组 KV',18)
text(28,365,'完整序列的位置对数：N(N+1)/2；单步读取与缓存量随 N 线性增长。',20)
text(28,400,'单 Head 示意，包含当前位置；固定模型配置，缓存全部历史。色块表示位置，不表示权重。',16)
save('context-costs')
start(330)
text(28,34,'同一个问题 + 相同文档集合',22);text(750,34,'只移动含答案的文档',20)
for y,k,label in [(70,0,'开头'),(145,3,'中间'),(220,6,'结尾')]:
 text(28,y+32,label,21)
 for j in range(7):
  box(115+j*123,y,111,48,'#dcece4' if j==k else '#f1f4f6')
  text(127+j*123,y+30,'相关文档' if j==k else '其他文档',18)
text(28,311,'比较不同位置的回答正确率；此图展示实验设置，不表示模型得分。',18)
save('context-evidence-position')
