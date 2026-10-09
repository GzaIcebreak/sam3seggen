"""A name imported inside a function is local to the WHOLE function, so a use of it that
comes before the import raises UnboundLocalError at run time -- which is what took down
a batch when a block that used `trimesh` was pasted above the block that imported it.
Static check over every module of the repo."""
import ast
import glob
import os
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_import_misuses(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    bad = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        imported = {}
        for node in ast.walk(func):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    imported[name] = min(imported.get(name, node.lineno), node.lineno)
        if not imported:
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                    and node.id in imported and node.lineno < imported[node.id]:
                bad.append(f"{os.path.basename(path)}:{node.lineno} uses {node.id!r} before "
                           f"its local import on line {imported[node.id]} in {func.name}()")
    return bad


class LocalImportTest(unittest.TestCase):
    def test_no_module_uses_a_locally_imported_name_before_importing_it(self):
        problems = []
        for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
            problems += local_import_misuses(path)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
