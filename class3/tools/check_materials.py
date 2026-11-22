"""检查课件时长、页数、字体覆盖、Notebook输出和本地链接。"""
import json
from pathlib import Path
import re
import zipfile

import nbformat
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]


def main():
    slides=json.loads((ROOT/'slides.json').read_text())
    assert sum(s['minutes'] for s in slides)==60
    assert sum(s['minutes'] for s in slides if s['section']=='复习')==25
    ppt=Presentation(ROOT/'class3_attention.pptx')
    assert len(ppt.slides)==len(slides)
    for page in ppt.slides:
        assert page.notes_slide.notes_text_frame.text.strip()
        for shape in page.shapes:
            if shape.has_text_frame and shape.text.strip():
                assert shape.text_frame.auto_size.name=='NONE'
            style=shape._element.find(qn('p:style'))
            if style is not None:
                assert style.find(qn('a:effectRef')).get('idx')=='0'
    notebook=nbformat.read(ROOT/'demo.ipynb',as_version=4)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert len(cells)==10 and all(c.execution_count is not None for c in cells)
    assert all(o.output_type!='error' for c in cells for o in c.outputs)
    for file in ROOT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if '://' in link or link.startswith('#'):continue
            path=link.split('#')[0]
            assert (file.parent/path).exists(),(file.name,link)
    # Validate actual preview typeface; fontTools is a Matplotlib dependency.
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    from presentation import Canvas
    from slide_design import course_fonts
    regular,bold,_=course_fonts()
    font=TTFont(regular,fontNumber=0)
    available=font.getBestCmap()
    metrics=json.loads((ROOT/'outputs/slide_previews/layout_report.json').read_text())
    missing=set()
    for page in metrics:
        for entry in page['text']:
            if entry['mono']:continue
            missing.update(ch for ch in entry['text'] if not ch.isspace() and ord(ch) not in available)
    assert not missing,missing
    print(f'教材一致性：{len(slides)}页/60分钟/25分钟复习、讲者备注、固定文本框、无主题阴影、')
    print('            10个已执行代码单元、本地链接与字体字符覆盖通过。')


if __name__=='__main__':
    main()
