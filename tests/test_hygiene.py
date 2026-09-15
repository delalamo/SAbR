import ast
from pathlib import Path

import sabr

SOURCE = Path(sabr.__file__).parent


def _walk_with_scope(node, inside_function=False, inside_class=False):
    for child in ast.iter_child_nodes(node):
        child_inside_function = inside_function or isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        child_inside_class = inside_class or isinstance(child, ast.ClassDef)
        yield child, inside_function, inside_class
        yield from _walk_with_scope(
            child, child_inside_function, child_inside_class
        )


def test_no_lazy_imports_or_module_variable_annotations():
    paths = list(SOURCE.glob("*.py"))
    assert paths, f"No installed source files found in {SOURCE}"
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node, inside_function, inside_class in _walk_with_scope(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert (
                    not inside_function
                ), f"Lazy import in {path}: {node.lineno}"
            if isinstance(node, ast.AnnAssign):
                assert (
                    inside_function or inside_class
                ), f"Module variable annotation in {path}: {node.lineno}"
