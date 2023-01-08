import ast
import os.path
import argparse

class NamesCollector(ast.NodeTransformer):
    """
    Traverses AST, removes docstrings and renames symbols (variables, funcion names, classes), e.g. "func1", "var2".
    """
    variables: dict
    functions: dict
    arguments: dict
    classes: dict
    calls: list

    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.classes = {}
        self.arguments = {}
        self.calls = [{}]

    def _fresh_name(self, node, attr_name, container, type):
        """Generates a fresh name for a symbol or takes a previously generated one.

        Args:
            node: current AST node
            attr_name: attribute containing symbol's name (e.g. "id", "name")
            container: dictionary {name: fresh_name}
            type: type of a symbol, e.g. "func", "arg", "var"
        """
        primary_name = getattr(node, attr_name)

        if primary_name in container:
            new_name = f"{type}{len(container)}"
        elif primary_name in self.calls[-1]:
            new_name = self.calls[-1][primary_name]
        else:
            new_name = primary_name

        container[primary_name] = new_name

    def _rename_node(self, node, attr_name, type, container):
        """Renames symbol node and traverses all of its' children.

        Args:
            node: current AST node
            attr_name: attribute containing symbol's name (e.g. "id", "name")
            container: dictionary {name: fresh_name}
            type: type of a symbol, e.g. "func", "arg", "var"

        Returns:
            New node of original type with generated name. 
        """
        self._fresh_name(node, attr_name, container, type)

        self.calls.append(self.arguments)
        self.generic_visit(node)
        self.calls.pop()

        new_name = container[getattr(node, attr_name)]

        # TODO: fails on type(ast.Name)
        return type(node)(**{**node.__dict__, f"{attr_name}": new_name})
    
    def _remove_docstring(self, node):
        if ast.get_docstring(node) != None and len(node):
            node.body = node.body[1:]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return self._rename_node(node, "id", "var", self.variables)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return self._rename_node(node, "arg", "arg", self.arguments)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._remove_docstring(node)
        return self._rename_node(node, "name", "func", self.functions)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self._remove_docstring(node)
        return self._rename_node(node, "name", "cls", self.classes)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self._remove_docstring(node)
        return self._rename_node(node, "name", "func", self.functions)


class ASTNormalizer:
    def __init__(self, input_file):
        source_code = open(input_file, "r").read()
        self.ast_tree = ast.parse(source_code)

    def normalize_source_file(self):
        transformed = NamesCollector().visit(self.ast_tree)
        return ast.unparse(transformed)


class Comparator:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def _calculate_levenstein_distance(self, s1: str, s2: str) -> float:
        n = len(s1)
        m = len(s2)

        f = [[0] * (m + 1) for _ in range(0, n)]

        for i in range(1, n):
            f[i][0] = i

        for j in range(1, m):
            f[0][j] = j

        for i in range(1, n):
            for j in range(1, m):
                w = 0 if s1[i - 1] == s2[j - 1] else 1
                mp = min(min(f[i - 1][j], f[i][j - 1]) +
                         1, f[i - 1][j - 1] + w)
                if f[i][j] != 0:
                    f[i][j] = min(f[i][j], mp)
                else:
                    f[i][j] = mp

        return f[n][m]

    def get_normalized_levenstein_distance(self, s1: str, s2: str) -> float:
        if len(s1) == 0 or len(s2):
            return 0

        return self._calculate_levenstein_distance(s1, s2) / max(len(s1), len(s2))

    def compare_files(self, file1, file2) -> float:
        normalized_file1 = ASTNormalizer(file1).normalize_source_file()
        normalized_file2 = ASTNormalizer(file2).normalize_source_file()
        return self.get_normalized_levenstein_distance(normalized_file1, normalized_file2)

    def start(self):
        with open(self.input_file, "r") as inF, open(self.output_file, "w") as ouF:
            for line in inF.readlines():
                f1, f2 = line.split()

                f1_exists = os.path.exists(f1)
                f2_exists = os.path.exists(f2)

                if not f1_exists:
                    ouF.write(f"File {f1} cannot be found. ")

                if not f2_exists:
                    ouF.write(f"File {f2} cannot be found. ")
                
                if not f1_exists or not f2_exists:
                    ouF.write("\n")
                    continue


                score = self.compare_files(f1, f2)
                ouF.write(f"{score}\n")


def main(input_file, output_file):
    cmp = Comparator(input_file, output_file)
    cmp.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compare",
                                     description="Calculates score of similarity for python source code files using Levenshtein distance.")
    parser.add_argument("input_file",  help="a file containing pairs of source code files to compare")
    parser.add_argument("output_file", help="a file containing socres of similarity for every pair from input files")
    args = parser.parse_args()

    main(args.input_file, args.output_file)