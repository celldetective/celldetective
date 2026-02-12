---
description: Docstring and Typing alignment
---

# Workflow: Autonomous Python Docstring & Type Alignment

**Role:** Expert Python Documentation Agent (NumPy/SciPy Style Specialist).
**Target Version:** Python 3.9+ (Strict Compatibility).
**Objective:** Audit codebase for docstring anomalies and missing type hints, then autonomously repair them while adhering to Python 3.9 constraints.

---

## Phase 0: The Audit (Anomaly & Type Detection)
**Action:** Execute the existing `docstring_auditor.py` script located at the project root.
**Command:** `python docstring_auditor.py ./celldetective`

**Reference Logic (For Agent Context Only):**
> *Use this block to understand the JSON output structure, specifically the `suggested_type` field.*

```python
import ast
import json
import os
import sys
import re

def parse_docstring_types(docstring):
    """
    Extracts {arg_name: type_description} from NumPy style docstrings.
    Matches lines like: "param_name : type_name"
    """
    if not docstring:
        return {}
    
    # Regex for NumPy style "param : type"
    # Captures "param_name" and "type_description" (e.g., "x : int, optional")
    param_pattern = re.compile(r'^\s*(\w+)\s*:\s*(.+)', re.MULTILINE)
    
    types = {}
    lines = docstring.split('\n')
    
    # Simple state machine to find 'Parameters' section
    in_param_section = False
    for line in lines:
        stripped = line.strip()
        # Detect Header
        if "Parameters" in line and len(lines) > lines.index(line)+1 and "---" in lines[lines.index(line)+1]:
            in_param_section = True
            continue
        # Detect End of Section
        if in_param_section and ("Returns" in line or "Yields" in line or "Raises" in line):
            break
            
        if in_param_section and stripped:
            match = param_pattern.match(line)
            if match:
                arg_name, arg_type = match.groups()
                # Clean up "int, optional" -> "int" (Agent will handle Optional logic via default values)
                clean_type = arg_type.split(',')[0].strip()
                types[arg_name] = clean_type
                
    return types

def analyze_node(node, filename):
    issues = []
    
    # 1. Extract Signature Args & Hints
    # returns list of tuples: (arg_name, has_type_hint)
    sig_args = []
    for a in node.args.args:
        if a.arg in ('self', 'cls'):
            continue
        has_hint = a.annotation is not None
        sig_args.append((a.arg, has_hint))
    
    # 2. Extract Docstring & Parse Types
    docstring = ast.get_docstring(node)
    doc_types = parse_docstring_types(docstring) 

    # 3. Check for Anomalies
    for arg_name, has_hint in sig_args:
        doc_type_str = doc_types.get(arg_name)
        
        # TYPE GAP: Docstring has type, Signature does not
        if doc_type_str and not has_hint:
            issues.append({
                "function": node.name,
                "line": node.lineno,
                "type": "missing_type_hint",
                "message": f"Arg '{arg_name}' documented as '{doc_type_str}' but lacks type hint.",
                "suggested_type": doc_type_str
            })
            
        # GHOST ARG: Code has arg, Docstring doesn't
        if arg_name not in doc_types and docstring:
             issues.append({
                "function": node.name,
                "line": node.lineno,
                "type": "arg_mismatch",
                "message": f"Argument '{arg_name}' missing from docstring."
            })

    return issues

def scan_directory(root_dir):
    report = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                except Exception:
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
```

---

## Phase 1: Planning
1.  **Ingest Report:** Parse `audit_report.json`.
2.  **Prioritize:** Focus on `missing_type_hint` issues first, as they require code changes, not just comment changes.
3.  **Plan Artifact:** Create a **Migration Plan** listing files to modify.

---

## Phase 2: Execution (The "Repair" Logic)
**CRITICAL: Python 3.9 Compatibility Rules**
You must strictly adhere to these typing rules to ensure support for Python 3.9 - 3.11.

1.  **NO Bitwise Union (`|`):**
    * ❌ Incorrect: `str | int` (Only valid in 3.10+)
    * ✅ Correct: `Union[str, int]` (Import `Union` from `typing`)
2.  **Generics (PEP 585):**
    * ✅ Correct: `list[int]`, `dict[str, int]`, `tuple[int]` (Valid in 3.9+)
    * ⚠️ Acceptable: `List[int]`, `Dict[str, int]` (Legacy, but valid)
3.  **Optional:**
    * ❌ Incorrect: `int | None`
    * ✅ Correct: `Optional[int]` (Import `Optional` from `typing`)

**Action Steps:**
1.  **Inject Types:** For every `missing_type_hint`, take the `suggested_type` from the report (e.g., "array_like") and convert it to a valid Python 3.9 type hint (e.g., `np.ndarray` or `list[float]`).
2.  **Import Management:** If you add `Union` or `Optional`, check if they are imported from `typing`. If not, add the import at the top of the file.
3.  **Docstring Sync:** If you update the code to have a default value (`x=None`), ensure the docstring says "optional" and the type hint becomes `Optional[T]`.

---

## Phase 3: Diátaxis & NumPy Formatting
* **Header Check:** Ensure "Parameters" and "Returns" headers exist.
* **Content:**
    ```python
    Parameters
    ----------
    arg_name : type
        Description...
    ```
* **LaTeX:** Wrap math in `$x$`.

---

## Phase 4: Verification
**Definition of Done:**
1.  **Re-Run Audit:** Execute `python docstring_auditor.py [file_path]`.
2.  **Zero Tolerance:** The JSON report for that file must return `[]`.
3.  **Lint Check:** Ensure no syntax errors (e.g., missing imports for `Union`).