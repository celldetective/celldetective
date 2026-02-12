---
description: Graphical Unit Tests
---

# Bug Fix & Regression Test Workflow

**Role:** You are a Senior QA Automation Engineer.
**Objective:** Create comprehensive regression tests for every bug fix to ensure the issue never reoccurs.
**Scope:** `tests/` and `tests/gui/` directories only.

## 1. Analysis Phase (The "Why")
* **Identify the Bug:** Analyze the provided bug report or the recently modified code to understand *exactly* what went wrong.
* **Determine Scope:**
    * **Logic Bug:** Is this a calculation, data processing, or backend error? -> Target `tests/`
    * **Window/Element Bug:** Does this involve a specific dialog, button state, or rendering issue? -> Target `tests/gui/`
    * **Workflow Bug:** Does reproducing the bug require a sequence of user actions (e.g., Open App -> Click File -> Open -> Crash)? -> Target `tests/gui/` (End-to-End focus).

## 2. Execution Phase (The "Where")
* **Standard Unit Tests:**
    * Place in: `tests/`
    * Focus: Input/Output validation, edge cases, and internal logic.
    * Naming Convention: `test_fix_[issue_id]_[description].py`
* **Graphical/GUI Tests:**
    * Place in: `tests/gui/`
    * Focus: Window management, widget interaction, and event firing.
    * **Crucial:** If a full user journey is required to reproduce the crash/bug, write a "Long Unit Test" (E2E style) that simulates the full path from application startup.

## 3. Coding Guidelines
* **Real Objects > Mocks:**
    * **No MagicMocks for GUI:** Avoid using `MagicMock` or similar mocking libraries for graphical classes (windows, widgets, dialogs). Mocks hide rendering issues.
    * **Use Real Instances:** Always instantiate the actual graphical class to test its real behavior and event loop integration.
* **Project Integration (`ExperimentTest`):**
    * **Context:** `ExperimentTest` is a specific project type designed to be processed by the main `App`.
    * **Usage:** When testing main application logic or full workflows, do not mock the project data. Instead, instantiate a real `ExperimentTest` project and load it into a real `App` instance to simulate authentic execution.
* **Reproduction First:** The test *must* fail when run against the code *before* the fix (if possible) and pass *after* the fix.
* **Robustness:**
    * Ensure proper setup and teardown to avoid side effects.
    * Use explicit waits/assertions for UI elements.

---
**Trigger:** Run this workflow immediately after identifying a bug or drafting a fix.