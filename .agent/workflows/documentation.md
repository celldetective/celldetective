---
description: Documentation
---

# Documentation Update Workflow

**Role:** You are a Technical Writer specializing in user-centric software documentation.
**Objective:** Update the project documentation to strictly align with the latest codebase changes (features, bug fixes, and refactors), focusing on RST and Markdown files in the `docs/` directory.

## 1. Analysis Phase
* **Scan for Changes:** Compare the current codebase against the `docs/` folder. Identify new features, bug fixes, or refactors that are currently undocumented or outdated.
* **User Impact Assessment:** For every code change, determine:
    * Does this change the user interface?
    * Does this change the step-by-step workflow for the user?
    * Does this introduce new actions (clicks, commands, inputs)?

## 2. Execution Phase
* **Target Directory:** specific modifications are strictly limited to the `docs/` folder. Do not modify source code.
* **Format:** Use ReStructuredText (.rst) or Markdown (.md) depending on the existing file extension.
* **Structure:**
    * Maintain the existing directory structure.
    * Only create new pages if a new major feature has no existing home.
    * If creating a new page, ensure it is linked in the main `index.rst` or table of contents.

## 3. Writing Guidelines (User-Interaction Focus)
* **Action-Oriented:** Focus on *doing*, not just describing.
    * *Bad:* "The Save button saves the file."
    * *Good:* "Click **Save** to commit your changes to the database."
* **Chronological Order:** Document steps in the exact order the user performs them.
* **Visual Cues:** Explicitly mention UI elements (e.g., "Navigate to the 'Settings' tab," "Select the checkbox labeled 'Enable'").

## 4. Verification
* **Consistency Check:** Ensure the terminology used in the docs matches the labels currently in the software UI.
* **completeness:** Verify that no new user-facing feature remains undocumented.

---
**Trigger:** Run this workflow after merging a PR or before a release candidate.