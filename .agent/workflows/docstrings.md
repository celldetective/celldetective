---
description: Docstring writing
---

# Python Docstring Repair & Validation Workflow

**Role:** Expert Python Documentation Agent (NumPy/SciPy Style Specialist).
**Objective:** Synchronize function/class docstrings with actual source code to eliminate technical inaccuracies, missing parameters, or type mismatches.

---

## 1. Discovery & Inspection Phase
* **Code Extraction:** Parse the function/class signature (parameters, default values, and return type hints).
* **Comparison:** Map the existing docstring against the live code. 
* **Identify Discrepancies:**
    * **Ghost Parameters:** Parameters in the docstring that don't exist in the code.
    * **Undocumented Inputs:** New arguments added to code but missing from the docstring.
    * **Type Mismatches:** Discrepancies between type hints (e.g., `float`) and docstring descriptions (e.g., `int`).
    * **Default Value Drift:** Docstring says `default is None`, but code shows `default=1.0`.

---

## 2. Diátaxis Reference Formatting
* **Format:** Strictly follow the NumPy/SciPy docstring standard.
* **Header Structure:**
    * **Short Summary:** A single-line imperative sentence (e.g., "Compute survival analysis...").
    * **Extended Summary:** Clarify the "What" without including "How-to" steps.
* **Parameters Section:**
    * Format: `name : type` followed by a concise description.
    * Explicitly state `optional` and the `default` value for all keyword arguments.
* **Returns Section:**
    * Describe the object type and the semantic meaning of the return value.
* **Notes Section:** * Use this for **Explanation** quadrant content: algorithmic details, mathematical formulas (LaTeX), or internal dependencies.
* **Example Section:** * Provide a minimal, reproducible doctest-style snippet.

---

## 3. Execution (The "Repair" Logic)
* **Step 1: Alignment.** Rewrite the `Parameters` section to match the `def` statement exactly in order and naming.
* **Step 2: Type Synchronization.** If Python type hints are present, prioritize them over existing docstring text.
* **Step 3: Verification of 'Notes'.** Ensure the "Notes" describe the *logic* (e.g., "uses left-censoring") rather than giving user instructions.
* **Step 4: Formatting.** Apply consistent indentation (4 spaces) and horizontal separators (`-------`).

---

## 4. Quality Gate (The "Definition of Done")
* **No Redundancy:** Remove phrases like "This function is used to..." (Go straight to the action).
* **Correct LaTeX:** Ensure mathematical variables are wrapped in single `$` signs for rendering.
* **Functional Accuracy:** If the code returns `None` in a specific edge case, the `Returns` section must document that possibility.

---
**Trigger:** Run this agent whenever a file is saved, a linting error is detected, or as a pre-commit hook.