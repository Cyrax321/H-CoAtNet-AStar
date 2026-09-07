#!/usr/bin/env python3
import ast
from pathlib import Path
src = Path("ablation study/ablation_study.py").read_text()
tree = ast.parse(src)

classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
print("classes:", list(classes.keys()))

def print_func(node, src):
    body = ast.get_source_segment(src, node)
    print(f"\n=== {node.name} ({node.lineno}) ===")
    for i, ln in enumerate(body.splitlines(), start=1):
        print(f"{i:3d}: {ln}")

for cname in ["AblationCoAtNet", "HierarchicalSE"]:
    cls = classes[cname]
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("forward","_select","_raw_importance","__init__","select_patches"):
                print_func(node, src)
        elif isinstance(node, ast.Assign):
            # docstring assign
            pass

# Also print generate_compare block that builds tex and writes files.
func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate_compare")
print_func(func, src)
