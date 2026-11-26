"""核对课件、可见文字、真实实验输出、字体覆盖与本地链接。"""
import json
from pathlib import Path
import re
import sys
import zipfile

import nbformat
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from content import slides
from layout import COURSE_ROOT
sys.path.insert(0,str(COURSE_ROOT/'class2/tools'))
from slide_design import course_fonts


def main():
    items=json.loads((ROOT/'slides.json').read_text())
    assert len(items)==28==len(slides())
    assert sum(item['minutes'] for item in items)==60
    assert all('core' not in item and 'appendix' not in item for item in items)
    deck=Presentation(ROOT/'class4_ffn_activation_sampling.pptx')
    assert len(deck.slides)==28
    assert abs(deck.slide_width/914400-13.333333)<1e-5
    regular,bold,_=course_fonts()
    font=TTFont(regular,fontNumber=0)
    available=font.getBestCmap();missing=set()
    metrics=json.loads((ROOT/'outputs/slide_previews/layout_report.json').read_text())
    for item,page,metric in zip(items,deck.slides,metrics):
        assert len(item['visible_text'])>=3,item['page']
        assert len(''.join(item['visible_text']))>=60,item['page']
        assert page.notes_slide.notes_text_frame.text.strip()
        assert page._element.find(qn('p:transition')) is not None
        for shape in page.shapes:
            assert shape.left>=-10000 and shape.top>=-10000
            assert shape.left+shape.width<=deck.slide_width+20000,(item['page'],shape.name)
            assert shape.top+shape.height<=deck.slide_height+20000,(item['page'],shape.name)
            if shape.has_text_frame and shape.text.strip():
                assert shape.text_frame.auto_size.name=='NONE'
            style=shape._element.find(qn('p:style'))
            if style is not None:assert style.find(qn('a:effectRef')).get('idx')=='0'
        for entry in metric['text']:
            if not entry['mono']:
                missing.update(ch for ch in entry['text'] if not ch.isspace() and ord(ch) not in available)
    assert not missing,missing
    visible='\n'.join(entry['text'] for page in metrics for entry in page['text'])
    assert not re.search(r'主线|深读|附录|精选路线|详细路线',visible)
    check=json.loads((ROOT/'outputs/slide_previews/validation.json').read_text())
    assert check=={'pages':28,'text_formula_overlap':0,'out_of_bounds':0},check
    notebook=nbformat.read(ROOT/'demo.ipynb',as_version=4);nbformat.validate(notebook)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    assert len(code_cells)==18
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(o.output_type=='error' for cell in code_cells for o in cell.outputs)
    for file in ROOT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (file.parent/target.split('#')[0]).exists(),(file.name,target)
    with zipfile.ZipFile(ROOT/'class4_ffn_activation_sampling.pptx') as z:
        links=z.read('ppt/slides/_rels/slide28.xml.rels').decode()
        assert 'arxiv.org' in links and 'https://' in links
    assert (ROOT/'class4_ffn_activation_sampling.pdf').stat().st_size>100000
    print('PASS: 28页连续课程 / 60分钟、16:9、原生形状边界、字体覆盖、讲者备注、淡入转场、')
    print('      零文字/公式重叠、18个已执行单元、文献超链接、本地文档链接。')


if __name__=='__main__':main()
