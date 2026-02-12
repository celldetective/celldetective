---
description: Docstring writing
---

# Workflow: Autonomous Python Docstring Alignment

**Role:** Expert Python Documentation Agent (NumPy/SciPy Style Specialist).
**Objective:** Proactively audit the codebase for docstring anomalies, plan a repair strategy, and synchronize function/class docstrings with actual source code to ensure semantic and syntactic accuracy.

---

## Phase 0: The Audit (Anomaly Detection)
**Action:** Execute the existing anomaly detection script located at the project root.
**Command:** `python docstring_auditor.py ./celldetective` (or target directory).

**Reference Logic (For Agent Context Only):**
> *Note: This script is already present at the root. Use this block only to understand the JSON output structure.*

```python
import ast
import json
import os
import sys

def analyze_node(node, filename):
    issues = []
    
    # 1. Extract Signature Args (excluding self/cls)
    sig_args = [a.arg for a in node.args.args if a.arg not in ('self', 'cls')]
    
    # 2. Extract Docstring
    docstring = ast.get_docstring(node)
    
    if not docstring:
        return [{
            "function": node.name,
            "line": node.lineno,
            "type": "missing_docstring",
            "message": "Function has no docstring."
        }]

    # 3. Check for NumPy Style Headers
    if sig_args and "Parameters" not in docstring:
        issues.append({
            "function": node.name,
            "line": node.lineno,
            "type": "style_violation",
            "message": "Missing 'Parameters' header in docstring."
        })

    # 4. Check for Argument Mismatches (The 'Ghost' Check)
    for arg in sig_args:
        if arg not in docstring:
            issues.append({
                "function": node.name,
                "line": node.lineno,
                "type": "arg_mismatch",
                "message": f"Argument '{arg}' exists in code but is missing from docstring."
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
                except (SyntaxError, UnicodeDecodeError):
                    continue

                file_issues = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        file_issues.extend(analyze_node(node, path))
                
                if file_issues:
                    report[path] = file_issues
    
    # Output JSON for the Agent to parse
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    scan_directory(target)
```

---

## Phase 1: Planning & Prioritization
1.  **Ingest Report:** Read the JSON output from Phase 0.
2.  **Triage:** Group issues by file. Prioritize files with "arg_mismatch" errors over simple "style_violation" errors.
3.  **Artifact Creation:** Generate a **Repair Plan** (Markdown list) summarizing the work to be done.
    * *Example:* "- [ ] `src/utils.py`: Fix 3 missing args in `calc_metric` and add Parameters header."

---

## Phase 2: Execution (The "Repair" Logic)
Iterate through the Repair Plan. For each file:

1.  **Deep Inspection:** Parse the function signature, including default values (`default=None`) and type hints (`-> float`).
2.  **Ghost Busting:** Identify parameters in the *old* docstring that no longer exist in the code and remove them.
3.  **Drift Correction:** If a default value in the code differs from the docstring text (e.g., text says "default is 5" but code is `default=10`), update the text to match the code.
4.  **Semantic Fill:** If a new argument is added (e.g., `tolerance`), do not just add a placeholder. Read the function body to understand how `tolerance` is used, then write a meaningful description.

---

## Phase 3: Diátaxis & NumPy Formatting Standards
Apply these rules to every docstring you touch:

* **Structure:**
    ```python
    """
    Single line summary (imperative verb).

    Extended summary covering the "Why" and "What".

    Parameters
    ----------
    arg_name : type
        Description of the argument.
    
    Returns
    -------
    int
        Description of the return value.
    """
    ```
* **Type Hints:** Prioritize the type hints found in the function signature over the old docstring.
* **LaTeX:** Ensure mathematical variables are wrapped in single `$` signs (e.g., `$x$`, `$\alpha$`).
* **Notes Section:** Use this for algorithmic details or complexity (Big O notation), not for basic usage instructions.

---

## Phase 4: Quality Gate (Verification)
**Definition of Done:**
1.  **Re-Run Audit:** Execute `python docstring_auditor.py [file_path]` on the specific file you just fixed.
2.  **Zero Tolerance:** The JSON report for that file must return an empty list `[]`.
3.  **Visual Check:** Ensure no linting errors are introduced (indentation, trailing whitespace).

**Trigger:**
* **Primary:** Run manually via command "Run Docstring Audit".
* **Secondary:** Run automatically when a comprehensive refactor is requested.