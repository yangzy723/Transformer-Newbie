"""用当前 Python 环境执行并保存 class1/demo.ipynb 的实际输出。

直接指定内核命令，避免注册或改动用户的全局 Jupyter kernelspec。
"""

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=root / "demo.ipynb",
                        help="可选：指定另一个需要执行的 Notebook")
    path = parser.parse_args().notebook.resolve()
    runtime = root / "outputs" / "jupyter_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(connection_file=str(runtime / "connection.json"))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=120,
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    print(f"已成功执行并保存 {len(code_cells)} 个代码单元。")


if __name__ == "__main__":
    main()
