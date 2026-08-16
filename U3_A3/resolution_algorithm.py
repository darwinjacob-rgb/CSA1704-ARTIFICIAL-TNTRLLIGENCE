"""
=====================================================================
 RESOLUTION ALGORITHM IMPLEMENTATION
 Course        : Artificial Intelligence (CSA17)
 Assessment    : Assessment Tool 3 - Annexure A
 Course Outcome: CO3 - Analyze knowledge representation and reasoning
                 using propositional logic, FOL, inference, resolution
                 and chaining methods.
 Description   : A generic Propositional-Logic Resolution engine that
                 converts a knowledge base + negated goal into CNF
                 clauses and repeatedly applies the Resolution Rule
                 until either the empty clause (proof found) or no
                 new clauses can be generated (proof fails).
=====================================================================
"""

from itertools import combinations


# ---------------------------------------------------------------------
# CORE RESOLUTION ENGINE
# ---------------------------------------------------------------------

def negate_literal(literal):
    """Return the logical complement of a literal, e.g. 'P' <-> '~P'."""
    return literal[1:] if literal.startswith('~') else '~' + literal


def clause_to_str(clause):
    """Pretty-print a clause (a frozenset of literals) as (A v B v ...)."""
    if not clause:
        return "{}  <-- NIL  (Empty Clause / contradiction)"
    return "(" + " v ".join(sorted(clause)) + ")"


def resolve(clause_i, clause_j):
    """
    Try to resolve two clauses on every possible pair of complementary
    literals. Returns the SET of all possible resolvent clauses
    (usually 0 or 1, but kept general).
    """
    resolvents = []
    for li in clause_i:
        if negate_literal(li) in clause_j:
            new_clause = (clause_i - {li}) | (clause_j - {negate_literal(li)})
            resolvents.append(frozenset(new_clause))
    return resolvents


def resolution_algorithm(kb_clauses, goal_clause, verbose=True):
    """
    Standard Resolution-Refutation Algorithm (Russell & Norvig style):

        1. clauses = CNF(KB) U CNF(~Goal)
        2. loop:
             for every pair (Ci, Cj) in clauses:
                 resolvents = RESOLVE(Ci, Cj)
                 if {} in resolvents: return "PROVED"
                 new = new U resolvents
             if new is subset of clauses: return "NOT PROVED"
             clauses = clauses U new
    """
    clauses = set(kb_clauses) | {goal_clause}
    step_no = 1
    trace = []

    if verbose:
        print("Initial Clause Set (KB clauses + Negated Goal):")
        for c in clauses:
            print("   ", clause_to_str(c))
        print()

    while True:
        new_clauses = set()
        pairs = list(combinations(clauses, 2))

        for (ci, cj) in pairs:
            resolvents = resolve(ci, cj)
            for r in resolvents:
                if verbose and r not in clauses and r not in new_clauses:
                    print(f"Step {step_no}: Resolve {clause_to_str(ci)}  "
                          f"and  {clause_to_str(cj)}")
                    print(f"        =>  {clause_to_str(r)}")
                    print()
                    trace.append((step_no, ci, cj, r))
                    step_no += 1

                if len(r) == 0:
                    return True, trace          # Empty clause -> proved

                new_clauses.add(r)

        if new_clauses.issubset(clauses):
            return False, trace                 # No new info -> not proved

        clauses |= new_clauses


# ---------------------------------------------------------------------
# HELPER TO RUN + DISPLAY ONE QUESTION
# ---------------------------------------------------------------------

def run_question(title, kb_english, kb_clauses, goal_english, goal_clause):
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("\nKnowledge Base (Propositional Logic):")
    for line in kb_english:
        print("   ", line)

    print(f"\nGoal: {goal_english}")
    print(f"Negated Goal (added as a clause): {clause_to_str(goal_clause)}\n")

    proved, trace = resolution_algorithm(kb_clauses, goal_clause)

    print("-" * 70)
    if proved:
        print(f"RESULT: Empty clause (NIL) derived -> GOAL IS PROVED TRUE.")
    else:
        print("RESULT: No empty clause derivable -> GOAL COULD NOT BE PROVED.")
    print("-" * 70 + "\n\n")
    return proved


# ---------------------------------------------------------------------
# THE 5 PROBLEMS FROM ANNEXURE A
# ---------------------------------------------------------------------

def main():

    # ---------------- Question 1: Rain and Wet Ground -----------------
    run_question(
        "QUESTION 1 : RAIN AND WET GROUND",
        kb_english=[
            "P  = 'It rains'",
            "Q  = 'The ground is wet'",
            "1. P -> Q      i.e. CNF: (~P v Q)",
            "2. P           i.e. CNF: (P)",
        ],
        kb_clauses=[frozenset({'~P', 'Q'}), frozenset({'P'})],
        goal_english="Q  ('The ground is wet')",
        goal_clause=frozenset({'~Q'}),
    )

    # --------------- Question 2: Student Assignment Submission --------
    run_question(
        "QUESTION 2 : STUDENT ASSIGNMENT SUBMISSION",
        kb_english=[
            "P  = 'Student submits the assignment'",
            "Q  = 'Student receives internal marks'",
            "1. P -> Q      i.e. CNF: (~P v Q)",
            "2. P (Rahul submitted)   i.e. CNF: (P)",
        ],
        kb_clauses=[frozenset({'~P', 'Q'}), frozenset({'P'})],
        goal_english="Q  ('Rahul receives internal marks')",
        goal_clause=frozenset({'~Q'}),
    )

    # --------------------- Question 3: Library Membership -------------
    run_question(
        "QUESTION 3 : LIBRARY MEMBERSHIP",
        kb_english=[
            "P  = 'Person is a library member'",
            "Q  = 'Person can borrow books'",
            "1. P -> Q      i.e. CNF: (~P v Q)",
            "2. P (Priya is a member) i.e. CNF: (P)",
        ],
        kb_clauses=[frozenset({'~P', 'Q'}), frozenset({'P'})],
        goal_english="Q  ('Priya can borrow books')",
        goal_clause=frozenset({'~Q'}),
    )

    # -------------------- Question 4: Placement Eligibility -----------
    run_question(
        "QUESTION 4 : PLACEMENT ELIGIBILITY",
        kb_english=[
            "P  = 'Student clears the aptitude test'",
            "Q  = 'Student is eligible for placement'",
            "1. P -> Q      i.e. CNF: (~P v Q)",
            "2. P (Arun cleared the test) i.e. CNF: (P)",
        ],
        kb_clauses=[frozenset({'~P', 'Q'}), frozenset({'P'})],
        goal_english="Q  ('Arun is eligible for placement')",
        goal_clause=frozenset({'~Q'}),
    )

    # -------------------- Question 5: Access Control System -----------
    run_question(
        "QUESTION 5 : ACCESS CONTROL SYSTEM",
        kb_english=[
            "P  = 'User enters the correct password'",
            "Q  = 'User is authenticated'",
            "R  = 'User is granted access'",
            "1. P -> Q      i.e. CNF: (~P v Q)",
            "2. Q -> R      i.e. CNF: (~Q v R)",
            "3. P (correct password entered) i.e. CNF: (P)",
        ],
        kb_clauses=[
            frozenset({'~P', 'Q'}),
            frozenset({'~Q', 'R'}),
            frozenset({'P'}),
        ],
        goal_english="R  ('User is granted access')",
        goal_clause=frozenset({'~R'}),
    )


if __name__ == "__main__":
    main()
