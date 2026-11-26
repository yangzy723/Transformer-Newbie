"""生成第四课原生可编辑 PPT、矢量文字 PDF、逐页文字及预览。"""
import argparse
import json
import math
from pathlib import Path
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement
from content import slides,SOURCES
from layout import ROOT,Canvas,header,save_vector_pdf,validate_layout
from ffn_pages import draw as draw_ffn
sys.path.insert(0,str(ROOT/'code'))
from lesson import build_report


def write_texts(items,canvases):
    body=['# 第四课：FFN、激活函数与采样\n',
          '28页，60分钟。以下按页列出PPT中的正文、公式与结论。\n']
    guide=['# 第四课逐页教案\n','28页，约60分钟，按页连续讲授。实验在对应内容处穿插。\n',
           '| 时间 | 页码 | 讲授内容 |\n|---|---|---|']
    minute=0
    for item in items:
        end=minute+item['minutes']
        guide.append(f"| {minute}–{end} | {item['page']} | {item['title']} |")
        minute=end
    guide += ['\n数值与代码对应 assets/lesson_report.json 和 demo.ipynb；课堂操作及补充说明列在各页教案中。\n']
    for item,c in zip(items,canvases):
        i=item['page']
        excluded={item['title'],item['subtitle'],item['takeaway'],item['section'],f'{i:02d}/{len(items):02d}'}
        visible=[m['text'] for m in c.text_metrics if not m['text'].startswith('TRANSFORMER  /') and m['text'] not in excluded]
        title=f"## {i:02d} · {item['title']}\n"
        body += ['\n'+title,item['subtitle']+'\n']+[t+'\n' for t in visible]
        formulas=getattr(c,'formula_expressions',[])
        body += ['$$\n'+expression+'\n$$\n' for expression in formulas]
        body += ['> '+item['takeaway']+'\n']
        guide += ['\n'+title,'**讲解要点**：'+item['notes']+'\n','**结论**：'+item['takeaway']+'\n']
        item['visible_text']=visible;item['formulas']=formulas
    for name,lines in [('slides.md',body),('teacher_guide.md',guide)]:
        (ROOT/name).write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'slides.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build(only=None):
    items=slides()
    report=build_report(ROOT/'assets/lesson_report.json')
    ppt=Presentation();ppt.slide_width=Inches(13.333333);ppt.slide_height=Inches(7.5)
    ppt.core_properties.title='第四课：FFN、激活函数与采样'
    ppt.core_properties.subject='FFN、非线性激活、LM Head 与概率采样'
    ppt.core_properties.author='Transformer Newbie'
    ppt.core_properties.keywords='Transformer, FFN, GELU, SwiGLU, Sampling'
    canvases=[];rendered=[];thumbs=[]
    out=ROOT/'outputs/slide_previews';out.mkdir(parents=True,exist_ok=True)
    for item in items:
        if only and item['page'] not in only:continue
        slide=ppt.slides.add_slide(ppt.slide_layouts[6]);c=Canvas(slide)
        header(c,item,len(items))
        try:
            if item['page']<=15:
                assert draw_ffn(c,item,report),item['kind']
            else:
                from output_pages import draw as draw_output
                assert draw_output(c,item,report),item['kind']
        except Exception:
            print(f"布局失败：第{item['page']}页 {item['title']}",flush=True)
            raise
        slide.notes_slide.notes_text_frame.text=(
            f"第{item['page']}页 · {item['section']}\n"+
            item['subtitle']+'\n\n'+item['notes']+'\n\n结论：'+item['takeaway']+
            '\n完整精度数值：assets/lesson_report.json。页面中的小数按显示位数取近似值。')
        transition=OxmlElement('p:transition');transition.set('spd','fast');transition.append(OxmlElement('p:fade'))
        slide._element.append(transition)
        c.image.save(out/f"slide_{item['page']:02d}.png")
        thumb=c.image.copy();thumb.thumbnail((628,354));thumbs.append(thumb)
        c.image=None;c.draw=None
        canvases.append(c);rendered.append(item)
        print(f"完成 {item['page']:02d}: {item['title']}",flush=True)
    validation=validate_layout(canvases)
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    metrics=[dict(page=item['page'],title=item['title'],text=c.text_metrics,ink=c.ink_boxes)
             for item,c in zip(rendered,canvases)]
    (out/'layout_report.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    suffix='_preview' if only else ''
    ppt.save(ROOT/f'class4_ffn_activation_sampling{suffix}.pptx')
    save_vector_pdf(canvases,ROOT/f'class4_ffn_activation_sampling{suffix}.pdf')
    sheet=Image.new('RGB',(1920,math.ceil(len(thumbs)/3)*374),'#DFE5EC')
    for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%3)*640+6,(i//3)*374+6))
    sheet.save(ROOT/'assets/slides_preview.jpg',quality=91)
    for start in range(0,len(thumbs),15):
        group=thumbs[start:start+15]
        contact=Image.new('RGB',(1920,math.ceil(len(group)/3)*374),'#DFE5EC')
        for j,thumb in enumerate(group):contact.paste(thumb,((j%3)*640+6,(j//3)*374+6))
        contact.save(out/f'contact_{start//15+1}.jpg',quality=92)
    if not only:
        write_texts(items,canvases)
        for page,name in [(1,'opening'),(2,'block'),(7,'expand'),(9,'ffn'),(11,'nonlinearity'),(14,'residual'),(17,'softmax'),(18,'sampling'),(20,'temperature'),(22,'top_p')]:
            import shutil
            shutil.copyfile(out/f'slide_{page:02d}.png',ROOT/f'assets/preview_{name}.png')
        import hashlib
        for path in (ROOT/'assets/formulas').glob('*.png'):
            if not any(path.stem==hashlib.sha256((expr+str(size)).encode()).hexdigest()[:12] for c in canvases for expr,size in getattr(c,'formula_specs',[])):
                path.unlink()
        from build_notebook import build as build_nb
        build_nb(ROOT)
    print(f'生成完成：{len(rendered)}页；文字/公式重叠与边界检查通过。',flush=True)
    return validation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages',help='仅渲染指定页，逗号分隔，用于版式检查')
    args=parser.parse_args()
    build(set(map(int,args.pages.split(','))) if args.pages else None)
