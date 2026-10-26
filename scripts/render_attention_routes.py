#!/usr/bin/env python3
"""Conceptual locations of changes; not a complete computation graph."""
from pathlib import Path
from html import escape
import gi
gi.require_version('GdkPixbuf','2.0')
from gi.repository import GdkPixbuf
out=Path(__file__).resolve().parents[1]/'book/assets/overview'
p=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">','<rect width="1120" height="650" fill="white"/>','<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#688091"/></marker></defs>','<g font-family="Noto Sans CJK SC, sans-serif" fill="#243342">']
def text(x,y,s,size=19):p.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(s)}</text>')
def rect(x,y,w,h,color='#eef3f6'):p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{color}" stroke="#c8d5dd"/>')
def line(x,y,x2,y2,arrow=True):p.append(f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="#688091" stroke-width="1.5"'+(' marker-end="url(#a)"' if arrow else '')+'/>')
text(28,32,'以 Softmax Attention 的匹配与汇总为参照',21)
for x,title,sub in [(28,'QKV 表示','生成用于匹配与汇总的向量'),(395,'位置匹配与权重','计算分数、Mask 与 Softmax'),(762,'Value 汇总与输出','按权重汇总，再组合 Head 输出')]:
 rect(x,195,330,90);text(x+17,229,title,22);text(x+17,261,sub,18)
line(360,240,390,240);line(729,240,757,240)
for x,rows in [(28,['KV 表示','MQA / GQA：共享 KV Head','MLA：压缩 KV 表示']),(395,['位置与连接','RoPE / ALiBi：引入位置信息','Sparse Attention：选择位置']),(762,['输出调节','Gated Attention','示例：对汇总结果施加 Gate'])]:
 rect(x,60,330,105,'#eaf3ee')
 for i,t in enumerate(rows):text(x+17,88+i*29,t,19 if i else 21)
 line(x+165,167,x+165,189)
text(28,320,'计算与存储实现',21)
rect(28,338,514,87);text(44,370,'FlashAttention',21);text(44,401,'分块执行 Attention，减少中间数据读写',19)
rect(568,338,524,87);text(584,370,'PagedAttention',21);text(584,401,'分页组织 KV Cache，按需分配与共享',19)
text(28,465,'另一种历史信息组织方式',21)
rect(28,485,1064,100,'#f2eff7')
text(46,519,'Linear Attention',21);text(275,519,'旧状态 + 当前输入 → 更新状态 → 读取输出',21)
text(46,555,'适用的核形式下，用状态累积历史；Hybrid Architectures 可组合不同序列模块。',19)
text(28,625,'标注表示方法主要改变的对象；RoPE 作用于 Q/K，ALiBi 作用于匹配分数。',17)
p.append('</g></svg>');path=out/'attention-design-locations.svg';path.write_text('\n'.join(p)+'\n');GdkPixbuf.Pixbuf.new_from_file(str(path)).savev(str(out/'attention-design-locations.png'),'png',[],[])
