"""Scan a source file for imports, attempt to map to distributions, and report installed versions.
Writes requirements_from_file.txt with package==version (when found).
Run from project root: python3 scripts/check_deps.py
"""
import ast
import os
import sys
from importlib import import_module
try:
    from importlib import metadata as importlib_metadata
except Exception:
    import importlib_metadata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_FILE = os.path.join(PROJECT_ROOT, 'receipt_scanning_through_ocr', 'ocr_more_lat.py')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'requirements_from_file.txt')

STANDARD_MODULES = set(sys.builtin_module_names) | {
    'os','sys','re','json','datetime','logging','time','traceback','functools',
    'collections','dataclasses','typing'
}


def parse_imports(path):
    with open(path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), path)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return sorted(imports)


# Best-effort mapping from top-level module -> distribution name
COMMON_MAP = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'skimage': 'scikit-image',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'pkg_resources': 'setuptools',
}


def resolve_distribution(module_name):
    # common map
    if module_name in COMMON_MAP:
        return COMMON_MAP[module_name]
    # try to use importlib_metadata to find distribution providing the top-level
    try:
        for dist in importlib_metadata.distributions():
            try:
                top = dist.read_text('top_level.txt')
            except Exception:
                top = None
            if top:
                for line in (top or '').splitlines():
                    if line.strip() == module_name:
                        return dist.metadata['Name']
    except Exception:
        pass
    # fallback to module_name as distribution
    return module_name


def get_version(dist_name, module_name):
    # Try import first
    try:
        import_module(module_name)
    except Exception:
        return None, f"Module '{module_name}' not importable"

    # Try distribution metadata
    try:
        v = importlib_metadata.version(dist_name)
        return v, None
    except Exception:
        # try distribution named as module
        try:
            v = importlib_metadata.version(module_name)
            return v, None
        except Exception as e:
            return None, str(e)


def main():
    imports = parse_imports(TARGET_FILE)
    third_party = [m for m in imports if m not in STANDARD_MODULES and not m.startswith('_')]

    results = []

    for mod in third_party:
        dist = resolve_distribution(mod)
        version, err = get_version(dist, mod)
        results.append((mod, dist, version, err))

    # Write requirements file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for mod, dist, version, err in results:
            if version:
                line = f"{dist}=={version}\n"
            else:
                line = f"# {dist}  # {err}\n"
            out.write(line)

    # Print summary
    print(f"Scanned: {TARGET_FILE}")
    for mod, dist, version, err in results:
        if version:
            print(f"{mod:15} -> {dist:25} == {version}")
        else:
            print(f"{mod:15} -> {dist:25} (not found)  reason: {err}")
    print('\nWrote requirements to:', OUTPUT_FILE)


if __name__ == '__main__':
    main()
