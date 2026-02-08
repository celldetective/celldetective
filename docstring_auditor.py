import ast
import json
import os
import sys
import re


def parse_docstring_types(docstring):
    """
    extracts {arg_name: type_description} from NumPy style docstrings.
    Matches lines like: "param_name : type_name"
    """
    if not docstring:
        return {}

    # Regex for NumPy style "param : type"
    # Captures "param_name" and "type_description"
    param_pattern = re.compile(r"^\s*(\w+)\s*:\s*(.+)", re.MULTILINE)

    types = {}
    lines = docstring.split("\n")

    # Simple state machine to find 'Parameters' section
    in_param_section = False
    for i, line in enumerate(lines):
        if "Parameters" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            in_param_section = True
            continue
        if in_param_section and line.strip() == "":
            # Empty line often ends the section in strict formatting,
            # but we'll keep reading until we hit another header or end of string
            continue
        if in_param_section and "Returns" in line:
            break

        if in_param_section:
            match = param_pattern.match(line)
            if match:
                arg_name, arg_type = match.groups()
                types[arg_name] = arg_type.strip()

    return types


def analyze_node(node, filename):
    issues = []

    # 1. Extract Signature Args & Types
    # returns list of tuples: (arg_name, has_type_hint)
    sig_args = []
    for a in node.args.args:
        if a.arg in ("self", "cls"):
            continue
        has_hint = a.annotation is not None
        sig_args.append((a.arg, has_hint))

    # 2. Extract Docstring & Parse Types
    docstring = ast.get_docstring(node)
    doc_types = parse_docstring_types(docstring)  # Returns {'arg': 'int', ...}

    # 3. Check for Typing Misalignments
    for arg_name, has_hint in sig_args:
        doc_type_str = doc_types.get(arg_name)

        # CASE A: Missing Type Hint (The "LLM Opportunity")
        # Docstring says "x : int", but code says "def func(x):"
        if doc_type_str and not has_hint:
            issues.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "type": "missing_type_hint",
                    "message": f"Arg '{arg_name}' is documented as '{doc_type_str}' but lacks a type hint in signature.",
                    "suggested_type": doc_type_str,
                }
            )

        # CASE B: Ghost Argument
        # Code has arg, Docstring doesn't mention it
        if arg_name not in doc_types and docstring:
            issues.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "type": "arg_mismatch",
                    "message": f"Argument '{arg_name}' exists in code but is missing from docstring.",
                }
            )

    return issues


def scan_directory(root_dir):
    report = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                except (SyntaxError, UnicodeDecodeError):
                    continue

                file_issues = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        file_issues.extend(analyze_node(node, path))

                if file_issues:
                    report[path] = file_issues

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    scan_directory(target)
