"""用当前 Python 执行课堂单元并保存实际输出，不启动外部内核。"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import argparse
import nbformat


def execute(path):
    path=Path(path).resolve()
    notebook=nbformat.read(path,as_version=4)
    namespace={'__name__':'__main__'}
    import os
    original=Path.cwd()
    try:
        os.chdir(path.parent)
        count=0
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            count+=1
            stream=StringIO()
            with redirect_stdout(stream):
                exec(compile(cell.source,str(path)+f':cell{count}','exec'),namespace)
            cell.execution_count=count
            cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else []
        nbformat.validate(notebook)
        nbformat.write(notebook,path)
    finally:os.chdir(original)
    print(f'已逐单元实际执行并保存 {count} 个代码单元：{path}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?',type=Path,default=Path(__file__).resolve().parents[1]/'demo.ipynb')
    execute(parser.parse_args().path)
