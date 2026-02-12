---
description: Diátaxis-Driven Documentation
---

# Diátaxis-Driven Documentation Workflow

**Role:** Senior Technical Writer & Diátaxis Architect.
**Objective:** Update project documentation (`docs/`) to align with codebase changes (features, fixes, refactors) while strictly maintaining the four distinct Diátaxis communication modes.

---

## 1. Analysis & Classification Phase
* **Scan for Changes:** Compare the current codebase against the `docs/` folder to find undocumented changes.
* **Diátaxis Mapping:** For every change, classify it into one or more of the following:
    * **Tutorials (Learning-oriented):** Does a beginner need a guided lesson to experience this?
    * **How-to Guides (Task-oriented):** Does this solve a specific "How do I...?" problem for a user with a goal?
    * **Reference (Information-oriented):** Is this a technical fact (API, CLI flag, UI label) that needs describing?
    * **Explanation (Understanding-oriented):** Does this require "big picture" context or a discussion of "why"?

---

## 2. Execution Phase (Quadrant Integrity)
* **Target Directory:** Modifications are strictly limited to the `docs/` folder.
* **Format:** Match existing `.rst` or `.md` extensions.
* **Structural Constraint:** You must respect the specific style of each quadrant:

### A. Tutorials (The Teacher)
* **Tone:** Encouraging and authoritative.
* **Rule:** Provide a single path to success. Do not offer options or "choose your own adventure" tangents.
* **Focus:** Getting the user started immediately.

### B. How-to Guides (The Recipe)
* **Tone:** Practical and efficient.
* **Rule:** Start with "How to..." or a goal-based title. Assume the user knows the basics.
* **Focus:** A sequence of steps to solve a specific, real-world problem.

### C. Reference (The Dictionary)
* **Tone:** Neutral, dry, and objective.
* **Rule:** Use tables, lists, and technical descriptions. No "how-to" instructions here.
* **Focus:** Completeness. List all parameters, return types, or UI elements.

### D. Explanation (The Historian)
* **Tone:** Discursive and descriptive.
* **Rule:** Explain the "Why" and the "How it works." Discuss architecture or design decisions.
* **Focus:** Building the user's mental model.

---

## 3. Writing & Formatting Guidelines
* **Separation of Concerns:** Never mix quadrants. Do not put "Explanation" (Theory) inside a "How-to" (Steps). 
* **Action-Oriented (Tutorials/How-tos):** Use the imperative mood (**"Click Save,"** **"Run the script"**).
* **Terminology:** Match software labels exactly. Use **Bold** for UI elements.
* **Links:** Link between quadrants. (e.g., A How-to Guide should link to a Reference page for parameter details).

---

## 4. Verification
* **Quadrant Check:** Does every updated file strictly serve only one of the four purposes?
* **Completeness:** Are all new user-facing features represented in **Reference**?
* **Linkage:** If a new page was created, is it added to the `index` or Table of Contents?

---
**Trigger:** Execute this workflow after code merges or during release preparation.