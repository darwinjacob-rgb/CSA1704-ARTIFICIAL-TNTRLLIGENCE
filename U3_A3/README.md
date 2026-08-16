# Resolution Algorithm Implementation
**Course:** Artificial Intelligence (CSA17) | **Assessment:** Assessment Tool 3 &ndash; Annexure A
**Course Outcome:** CO3 &ndash; Analyze knowledge representation and reasoning using propositional
logic, FOL, inference, resolution and chaining methods (BL4)

## Overview

This project implements the **Propositional Logic Resolution Algorithm** and applies it to
five knowledge-representation problems given in Annexure A:

| # | Scenario | Goal |
|---|----------|------|
| 1 | Rain and Wet Ground | Prove the ground is wet |
| 2 | Student Assignment Submission | Prove Rahul receives internal marks |
| 3 | Library Membership | Prove Priya can borrow books |
| 4 | Placement Eligibility | Prove Arun is eligible for placement |
| 5 | Access Control System | Prove the user is granted access |

For each scenario the program:
1. Formalizes the knowledge base in propositional logic.
2. Converts every implication to Conjunctive Normal Form (CNF) using `A → B ≡ (¬A ∨ B)`.
3. Negates the goal and adds it to the clause set.
4. Repeatedly resolves clause pairs containing complementary literals.
5. Reports the derivation of the Empty Clause (**NIL**) as proof that the goal is entailed.

## Repository Contents

| File | Description |
|------|-------------|
| `Problem_Statement.pdf` | All 5 questions, knowledge bases, goals and tasks, consolidated in one document. |
| `Solution.pdf` | Full hand-worked, step-by-step resolution proofs (CNF conversion → resolution steps → conclusion) for all 5 questions. |
| `resolution_algorithm.py` | Python implementation of a generic resolution-refutation engine, applied to all 5 problems. |
| `output.png` | Terminal-style screenshot of the program's console output for all 5 questions. |
| `Report.pdf` | Professional technical report: objective, theory, methodology, results table, discussion and conclusion. |
| `README.md` | This file. |

## How the Algorithm Works

Clauses are represented as Python `frozenset` objects of literal strings, e.g.
`{'~P', 'Q'}` represents the clause `(¬P ∨ Q)`.

```python
def resolve(clause_i, clause_j):
    """Resolve two clauses on every complementary literal pair."""
    resolvents = []
    for li in clause_i:
        if negate_literal(li) in clause_j:
            new_clause = (clause_i - {li}) | (clause_j - {negate_literal(li)})
            resolvents.append(frozenset(new_clause))
    return resolvents
```

The main loop (`resolution_algorithm`) repeatedly resolves every pair of clauses in the
current set. If the **empty clause** (`frozenset()`) is ever produced, the goal is proved.
If a full pass produces no clause that wasn't already known, the goal cannot be proved from
the given knowledge base.

## How to Run

Requires Python 3 (no external libraries needed for the core algorithm).

```bash
python3 resolution_algorithm.py
```

This prints, for each of the 5 questions:
- The knowledge base in propositional logic and CNF.
- The negated goal.
- Every resolution step (which two clauses were resolved and the resulting clause).
- The final verdict (`GOAL IS PROVED TRUE`) once the Empty Clause (NIL) is derived.

## Sample Result (Question 5 &ndash; Access Control System)

```
Knowledge Base:
  C1: (¬P ∨ Q)      # correct password → authenticated
  C2: (¬Q ∨ R)      # authenticated → granted access
  C3: (P)           # correct password entered
  C4: (¬R)          # negated goal

Step 1: Resolve C1, C3  →  C5: (Q)
Step 2: Resolve C5, C2  →  C6: (R)
Step 3: Resolve C6, C4  →  NIL   (Empty Clause)

RESULT: GOAL PROVED — the user is granted access.
```

## Summary of Results

All five goals were proved by refutation — the Empty Clause (NIL) was successfully derived
in every case. Questions 1&ndash;4 (single implication, Modus-Ponens pattern) each took
**2 resolution steps**; Question 5 (a two-implication chain, P→Q→R) took **3 resolution
steps**, since it required deriving the intermediate fact before the final goal.

## Author

Prepared as part of SIMATS Engineering, Assessment Tool 3, Course CSA17 (Artificial
Intelligence).
