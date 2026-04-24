import os

def generate_tree(startpath, depth=None):
    prefix = '│   '
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if depth is not None and level > depth:
            continue
        indent = '│   ' * level
        print(f"{indent}├── {os.path.basename(root)}/")
        sub_indent = '│   ' * (level + 1)
        for f in files:
            print(f"{sub_indent}├── {f}")

generate_tree('.', depth=3)
