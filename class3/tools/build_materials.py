"""用同一份计算结果，生成课件、PDF、预览与文字教材。"""
import argparse
import json
import math
from pathlib import Path
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from presentation import ROOT,Canvas,header,reference,save_vector_pdf,validate_layout
from pages import draw

sys.path.insert(0,str(ROOT/'code'))
from experiments import build_report

FORMULAS={
'old_attention':r'\mathbf z=\sum_j a_j\mathbf v_j',
'embedding':r'\mathbf X=\mathbf E[I]',
'positions':r'\theta_{p,i}=p\omega_i',
'qkv_roles':r'\mathbf Q=\mathbf X\mathbf W_Q,\quad\mathbf K=\mathbf X\mathbf W_K,\quad\mathbf V=\mathbf X\mathbf W_V',
'one_query':r's_{2,j}=\mathbf q_2\cdot\mathbf k_j',
'score_matrix':r'\mathbf S_{\rm raw}=\mathbf Q\mathbf K^\mathsf{T}',
'scale':r'\mathbf S=\frac{\mathbf Q\mathbf K^\mathsf{T}}{\sqrt{d_k}}',
'mask':r'\mathbf A=\operatorname{softmax}_j(\mathbf S+\mathbf M)',
'mix':r'\mathbf Z=\mathbf A\mathbf V',
'change_v':r'\Delta\mathbf z_i=A_{i,0}\Delta\mathbf v_0',
'multihead':r'\operatorname{MultiHead}(\mathbf X)=\operatorname{Concat}(\mathbf Z^{(1)},\ldots,\mathbf Z^{(H)})\mathbf W_O',
}
CODE={
'qkv_roles':['q, k, v = project(x)'],
'projections':['q = x @ w_q','k = x @ w_k','v = x @ w_v'],
'one_query':['scores = q[-1] @ k.T / math.sqrt(2)','weights = scores.softmax(-1)'],
'score_matrix':['raw_scores = q @ k.T'],
'scale':['scores = raw_scores / math.sqrt(q.shape[-1])'],
'mask':['future = torch.ones(3, 3, dtype=torch.bool).triu(1)','masked = scores.masked_fill(future, -torch.inf)'],
'mix':['weights = masked.softmax(dim=-1)','output = weights @ v'],
'change_v':['changed_v = v.clone()','changed_v[0] += v.new_tensor([2., 0.])','changed = attention(q, k, changed_v)'],
'change_k':['changed_k = k.clone()','changed_k[0] += k.new_tensor([1., 0.])','changed = attention(q, changed_k, v)'],
'causality':['changed_x = x.clone()','changed_x[-1] += x.new_tensor([3., -2.])','q2, k2, v2 = project(changed_x)'],
'core_code':['q, k, v = x @ w_q, x @ w_k, x @ w_v',
             'scores = q @ k.T / math.sqrt(q.shape[-1])',
             'future = torch.ones(3, 3, dtype=torch.bool).triu(1)',
             'masked = scores.masked_fill(future, -torch.inf)',
             'weights = masked.softmax(dim=-1)','output = weights @ v'],
}


def write_texts(items):
    slides=['# 第三课：重新走进 Transformer，理解 Attention\n']
    guide=['# 第三课教案\n','60分钟，21页。前25分钟复习，后半节完整计算单头因果Attention。\n',
           '课前打开PPT与Notebook，运行 §python class3/tools/check_experiments.py§。数值统一来自 §assets/lesson_report.json§。\n']
    mapping=['# 第三课知识与代码对照\n',
             '代码使用小写变量，公式使用大写矩阵；位置从0开始，第三行是位置2。\n',
             '| 页 | 时间 | 知识 | Notebook实验 | 源码 |\n|---|---|---|---|---|']
    minute=0
    for n,item in enumerate(items,1):
        start,end=minute,minute+item['minutes'];minute=end
        title=f"{n:02d}．{item['title']}（{start}–{end}分钟）"
        slides += ['\n## '+title+'\n',item['subtitle'],'\n'.join('- '+s for s in item['explanation']),
                   '\n> '+item['takeaway'],'\n**讲者备注**：'+item['notes']]
        guide += ['\n## '+title+'\n','\n'.join('- '+s for s in item['explanation']),
                  '\n**讲法、提问与预期回答**：'+item['notes'],'\n**页面收束**：'+item['takeaway']]
        lab=f"实验{item['experiment']:02d}" if item['experiment'] else '图示/检查'
        guide += [f"\n**对应操作**：{lab}；[{item['source']}]({item['source']})。"]
        mapping += [f"| {n} | {start}–{end} | {item['title']} | {lab} | [{item['source']}]({item['source']}) |"]
        if item['kind'] in FORMULAS:
            slides += ['\n$$\n'+FORMULAS[item['kind']]+'\n$$']
        if item['kind'] in CODE:
            slides += ['\n§§§python\n'+'\n'.join(CODE[item['kind']])+'\n§§§']
    mapping += ['\n## 核心计算逐项对应\n']
    for kind in ('projections','score_matrix','scale','mask','mix','change_v','change_k','causality'):
        it=next(i for i in items if i['kind']==kind)
        mapping += ['\n### '+it['title']+'\n',it['takeaway']]
        if kind in FORMULAS:mapping += ['\n$$\n'+FORMULAS[kind]+'\n$$']
        mapping += ['\n§§§python\n'+'\n'.join(CODE[kind])+'\n§§§']
    mapping += [r'''
## 形状与接口约定

主例 $T=3,D=d_k=d_v=2$，省略batch维。Q/K/V为[3,2]，分数与权重为[3,3]，输出为[3,2]。
矩阵行是query位置，列是key位置，Softmax沿最后一维执行。

数学使用X@W。nn.Linear.weight存为[out,in]，计算用X@weight.T+bias。
第20页是独立的多头结构示意：B=1,T=3,D=4,H=2,d_h=2。

RoPE在投影Q/K之后、点积之前。实际接口复用第二课positions.apply_rope；V不旋转，
干预前后位置保持一致。主手算先分离位置旋转，以便逐项核算矩阵。

只改K的本例中，第一行只可读自身；第二行Q与改变量正交，所以前两行权重不变，第三行改变。
BPE字符演示与真实ByteLevel配置使用各自的片段和ID。完整精度数值见assets/lesson_report.json。
''']
    guide += ['\n## 现场节奏\n',
              '复习若超时，保留完整矩阵、mask和修改未来输入实验。缩放方差推导留作追问，多头只读结构图。',
              '\n## 参考资料\n',
              '- [原始Transformer §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)\n- [RoFormer](https://arxiv.org/abs/2104.09864)\n- [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)']
    for name,body in (('slides.md',slides),('teacher_guide.md',guide),('knowledge_code_map.md',mapping)):
        (ROOT/name).write_text('\n'.join(body).replace('§',chr(96)).strip()+'\n',encoding='utf-8')


def build():
    items=json.loads((ROOT/'slides.json').read_text())
    report=build_report(ROOT/'assets/lesson_report.json')
    ppt=Presentation();ppt.slide_width=Inches(13.333333);ppt.slide_height=Inches(7.5)
    ppt.core_properties.title='第三课：重新走进 Transformer，理解 Attention'
    ppt.core_properties.subject='60分钟，前两课复习与单头因果注意力'
    ppt.core_properties.author='Transformer Newbie'
    canvases=[];minute=0;metrics=[]
    for n,item in enumerate(items,1):
        start,end=minute,minute+item['minutes'];minute=end
        slide=ppt.slides.add_slide(ppt.slide_layouts[6]);c=Canvas(slide)
        header(c,item,n,start,end,len(items));draw(c,item,report);reference(c,item)
        slide.notes_slide.notes_text_frame.text=(
            f"第{n}页 {start}–{end}分钟\n{item['title']}\n"+
            '\n'.join(item['explanation'])+'\n讲解与问题：'+item['notes']+
            f"\n代码：{item['source']}；Notebook实验：{item.get('experiment')}"+
            '\n数值来源：assets/lesson_report.json。完整精度计算，页面取近似值。')
        canvases.append(c);metrics.append(dict(page=n,title=item['title'],text=c.text_metrics,ink=c.ink_boxes))
    assert minute==60
    validation=validate_layout(canvases)
    ppt.save(ROOT/'class3_attention.pptx')
    save_vector_pdf(canvases,ROOT/'class3_attention.pdf')
    write_texts(items)
    out=ROOT/'outputs/slide_previews';out.mkdir(parents=True,exist_ok=True)
    (out/'layout_report.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    sheet=Image.new('RGB',(1920,math.ceil(len(items)/3)*374),'#DCE3EC')
    for i,c in enumerate(canvases):
        c.image.save(out/f'slide_{i+1:02d}.png')
        thumb=c.image.copy();thumb.thumbnail((628,354))
        sheet.paste(thumb,((i%3)*640+6,(i//3)*374+6))
    sheet.save(ROOT/'assets/slides_preview.jpg',quality=93)
    for name,i in (('architecture',1),('scores',11),('causality',17),('projection',9),('positions',7)):
        canvases[i].image.save(ROOT/f'assets/preview_{name}.png')
    print(f"课件完成：{len(items)}页，{minute}分钟；文字/公式边界检查通过。")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--slides-only',action='store_true')
    args=parser.parse_args()
    build()
    if not args.slides_only:
        from build_notebook import build as notebook
        print('Notebook模板：',notebook(ROOT))
