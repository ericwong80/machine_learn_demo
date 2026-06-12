from __future__ import annotations

import inspect
import os
from datetime import datetime


def create_output_dir() -> str:
    """
    创建图片输出目录。

    输出结构示例：
        当前脚本目录/output/knn_iris_learning/20260526_153000/
    """
    caller_frame = inspect.currentframe()
    caller_file = None
    if caller_frame is not None and caller_frame.f_back is not None:
        caller_file = caller_frame.f_back.f_globals.get("__file__")

    script_file = caller_file or __file__
    script_dir = os.path.dirname(os.path.abspath(script_file))
    module_name = os.path.splitext(os.path.basename(script_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(script_dir, "output", module_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir
