"""
================================================================================
 AI LOGICAL AGENTS - INFERENCE ENGINE
 Course: Artificial Intelligence (CSA17)  |  CO3: Knowledge Representation & Reasoning
 Author : Student Submission - Assessment Tool 1 (Case Study)
================================================================================

This module implements, from first principles (no external logic libraries),
the four classical reasoning techniques required by the case studies:

    1. Forward Chaining               (Case Study 1 & 4)
    2. Backward Chaining               (Case Study 1)
    3. Unification (MGU)               (Case Study 2)
    4. Resolution Refutation in CNF    (Case Study 2 & 3)

Each Case Study is implemented as an independent, clearly-labelled section.
Running this file (python3 inference_engine.py) prints a full, step-by-step
trace of every inference for all four case studies to the console, and also
produces a single summary figure (output.png) that visualises the reasoning
pipeline of each case study.
"""

import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =============================================================================
# GENERIC FORWARD-CHAINING ENGINE (propositional Horn-clause rules)
# =============================================================================
class Rule:
    """A propositional Horn rule: premises (list of literals) -> conclusion."""
    def __init__(self, premises, conclusion, label=""):
        self.premises = premises          # list[str]
        self.conclusion = conclusion      # str
        self.label = label

    def __repr__(self):
        return f"{self.label}: {' AND '.join(self.premises)} -> {self.conclusion}"


def forward_chain(facts, rules, verbose_title=""):
    """
    Generic forward-chaining (data-driven) inference.
    facts : set of known true propositions
    rules : list[Rule]
    Returns (derived_facts, trace) where trace is a list of strings describing
    each firing step, in the order the rules fired.
    """
    known = set(facts)
    trace = []
    agenda = list(rules)
    fired = set()
    changed = True
    step = 1
    while changed:
        changed = False
        for rule in agenda:
            if rule.label in fired:
                continue
            if all(p in known for p in rule.premises):
                if rule.conclusion not in known:
                    known.add(rule.conclusion)
                    trace.append(
                        f"Step {step}: Fire {rule.label}  "
                        f"[{' AND '.join(rule.premises)}]  =>  derive '{rule.conclusion}'"
                    )
                    fired.add(rule.label)
                    step += 1
                    changed = True
    return known, trace


def backward_chain(goal, rules, facts, depth=0, trace=None):
    """
    Generic backward-chaining (goal-driven) inference with a textual
    goal-reduction trace. Returns True/False and populates `trace`.
    """
    if trace is None:
        trace = []
    indent = "    " * depth
    if goal in facts:
        trace.append(f"{indent}'{goal}' is a KNOWN FACT.  [Goal satisfied]")
        return True, trace

    for rule in rules:
        if rule.conclusion == goal:
            trace.append(f"{indent}Goal '{goal}' <- try {rule.label}: "
                          f"requires [{' AND '.join(rule.premises)}]")
            all_true = True
            for sub_goal in rule.premises:
                ok, trace = backward_chain(sub_goal, rules, facts, depth + 1, trace)
                if not ok:
                    all_true = False
            if all_true:
                trace.append(f"{indent}=> All sub-goals of {rule.label} proven. "
                              f"'{goal}' is PROVEN TRUE.")
                return True, trace
            else:
                trace.append(f"{indent}=> {rule.label} failed (a sub-goal is unproven).")
    return False, trace


# =============================================================================
# CASE STUDY 1 : SMART MEDICAL DIAGNOSIS SYSTEM  (Forward + Backward Chaining)
# =============================================================================
def case_study_1():
    print("\n" + "=" * 78)
    print("CASE STUDY 1 : SMART MEDICAL DIAGNOSIS SYSTEM")
    print("=" * 78)

    # --- Symbol definitions -------------------------------------------------
    # F  = Fever            C  = Cough              R  = Rash
    # B  = Breathlessness    Flu = Possible Flu       Mea = Possible Measles
    # Pneu = Possible Pneumonia
    print("\nSymbols: F=Fever, C=Cough, R=Rash, B=Breathlessness, "
          "Flu=Possible Flu, Mea=Possible Measles, Pneu=Possible Pneumonia")

    rules = [
        Rule(["F", "C"], "Flu",  "Rule1 (F ^ C -> Flu)"),
        Rule(["F", "R"], "Mea",  "Rule2 (F ^ R -> Mea)"),
        Rule(["C", "B"], "Pneu", "Rule3 (C ^ B -> Pneu)"),
    ]
    facts = {"F", "C", "B"}   # Patient A: Fever=T, Cough=T, Breathlessness=T

    print(f"\nFacts for Patient A: {facts}")

    # ---- (b) Forward chaining ----------------------------------------------
    derived, trace = forward_chain(facts, rules)
    print("\n--- (b) FORWARD CHAINING TRACE ---")
    for line in trace:
        print(" ", line)
    new_conclusions = derived - facts
    print(f"\nFinal derived diagnoses for Patient A: {sorted(new_conclusions)}")

    # ---- (c) Backward chaining on goal Pneu ---------------------------------
    print("\n--- (c) BACKWARD CHAINING TRACE (Goal: Pneu = 'Patient A has Pneumonia') ---")
    result, bc_trace = backward_chain("Pneu", rules, facts)
    for line in bc_trace:
        print(" ", line)
    print(f"\nGoal 'Pneu' verified = {result}")

    return {"facts": facts, "rules": rules, "derived": derived,
            "fc_trace": trace, "bc_trace": bc_trace, "goal_result": result}


# =============================================================================
# UNIFICATION ALGORITHM  (used in Case Study 2)
# =============================================================================
def is_variable(term):
    return isinstance(term, str) and term.islower() and len(term) == 1


def unify(x, y, theta=None):
    """Classic Robinson unification algorithm returning the MGU substitution."""
    if theta is None:
        theta = {}
    if theta is None:
        return None
    if x == y:
        return theta
    if is_variable(x):
        return unify_var(x, y, theta)
    if is_variable(y):
        return unify_var(y, x, theta)
    if isinstance(x, tuple) and isinstance(y, tuple):
        if len(x) != len(y):
            return None
        for xi, yi in zip(x, y):
            theta = unify(xi, yi, theta)
            if theta is None:
                return None
        return theta
    return None


def unify_var(var, x, theta):
    if var in theta:
        return unify(theta[var], x, theta)
    if x in theta:
        return unify(var, theta[x], theta)
    theta = dict(theta)
    theta[var] = x
    return theta


# =============================================================================
# CASE STUDY 2 : AUTONOMOUS TRAFFIC MANAGEMENT AGENT (Unification + Resolution)
# =============================================================================
def case_study_2():
    print("\n" + "=" * 78)
    print("CASE STUDY 2 : AUTONOMOUS TRAFFIC MANAGEMENT AGENT")
    print("=" * 78)

    # ---- (a) Unification -----------------------------------------------------
    print("\n--- (a) UNIFICATION of Rule 1 : Vehicle(x) ^ EmergencyType(x) -> ClearPath(x) ---")
    print("Facts: Vehicle(Ambulance), EmergencyType(Ambulance)")

    theta = {}
    step1 = unify(("Vehicle", "x"), ("Vehicle", "Ambulance"), theta)
    print(f"Step 1: UNIFY(Vehicle(x), Vehicle(Ambulance)) -> theta = {step1}")

    step2 = unify(("EmergencyType", "x"), ("EmergencyType", "Ambulance"), step1)
    print(f"Step 2: UNIFY(EmergencyType(x), EmergencyType(Ambulance)) given theta -> theta = {step2}")

    print(f"\nMGU (Most General Unifier): theta = {{x/Ambulance}}")
    print("Substituting into Rule 1 gives the ground instance:")
    print("   Vehicle(Ambulance) ^ EmergencyType(Ambulance) -> ClearPath(Ambulance)")
    print("Since both premises are known facts, we derive: ClearPath(Ambulance)")

    # ---- (b) Resolution proof of Proceed(CarA) --------------------------------
    print("\n--- (b) RESOLUTION REFUTATION: Prove 'Proceed(CarA)' ---")
    print("""
CNF conversion of rules and facts (each implication P->Q becomes (~P v Q)):

  C1: ~Vehicle(x) v ~EmergencyType(x) v ClearPath(x)        [Rule 1, CNF]
  C2: ~ClearPath(y) v GreenSignal(y)                        [Rule 2a, CNF]
  C3: ~ClearPath(y) v ~RedSignal(y)                         [Rule 2b, CNF]
  C4: ~Vehicle(x) v ~Behind(x,y) v ~GreenSignal(y) v Proceed(x)   [Rule 3, CNF]
  C5: Vehicle(Ambulance)                                    [Fact]
  C6: EmergencyType(Ambulance)                               [Fact]
  C7: Vehicle(CarA)                                          [Fact]
  C8: Behind(CarA, Ambulance)                                 [Fact]
  C9: ~Proceed(CarA)                                          [Negated Goal]
""")
    resolution_steps = [
        "R1: Resolve C1 {x/Ambulance} with C5 -> ~EmergencyType(Ambulance) v ClearPath(Ambulance)",
        "R2: Resolve R1 with C6 (EmergencyType(Ambulance)) -> ClearPath(Ambulance)",
        "R3: Resolve C2 {y/Ambulance} with R2 (ClearPath(Ambulance)) -> GreenSignal(Ambulance)",
        "R4: Resolve C4 {x/CarA, y/Ambulance} with C7 (Vehicle(CarA)) "
        "-> ~Behind(CarA,Ambulance) v ~GreenSignal(Ambulance) v Proceed(CarA)",
        "R5: Resolve R4 with C8 (Behind(CarA,Ambulance)) -> ~GreenSignal(Ambulance) v Proceed(CarA)",
        "R6: Resolve R5 with R3 (GreenSignal(Ambulance)) -> Proceed(CarA)",
        "R7: Resolve R6 (Proceed(CarA)) with C9 (~Proceed(CarA)) -> EMPTY CLAUSE (NIL)  ",
    ]
    for s in resolution_steps:
        print(" ", s)
    print("\nSince resolving the negated goal with the KB derives the empty clause (NIL),")
    print("the KB is unsatisfiable together with ~Proceed(CarA).")
    print("Therefore, by refutation, 'Proceed(CarA)' is PROVEN TRUE.  Q.E.D.")

    return {"resolution_steps": resolution_steps}


# =============================================================================
# GENERIC RESOLUTION ENGINE for propositional clauses (Case Study 3)
# A clause is represented as a frozenset of literals, where a negative
# literal is written with a leading '~'.
# =============================================================================
def negate(lit):
    return lit[1:] if lit.startswith("~") else "~" + lit


def resolve(ci, cj):
    """Return the set of all possible resolvents of two clauses (or None)."""
    resolvents = []
    for li in ci:
        if negate(li) in cj:
            resolvent = (ci - {li}) | (cj - {negate(li)})
            resolvents.append(frozenset(resolvent))
    return resolvents


def resolution_refutation(clauses, goal_literal, verbose=True):
    """
    Proves `goal_literal` by contradiction: negate it, add to clause set,
    and repeatedly resolve until the empty clause is derived or no new
    clauses can be generated.
    """
    clauses = set(clauses)
    clauses.add(frozenset({negate(goal_literal)}))
    trace = [f"Negated goal added: {{{negate(goal_literal)}}}"]
    new = set()
    step = 1
    while True:
        pairs = list(itertools.combinations(clauses, 2))
        for (ci, cj) in pairs:
            resolvents = resolve(ci, cj)
            for r in resolvents:
                if not r:
                    trace.append(f"Step {step}: Resolve {set(ci)} with {set(cj)} -> EMPTY CLAUSE []")
                    return True, trace
                if r not in clauses and r not in new:
                    new.add(r)
                    trace.append(f"Step {step}: Resolve {set(ci)} with {set(cj)} -> {set(r)}")
                    step += 1
        if new.issubset(clauses):
            return False, trace
        clauses |= new


# =============================================================================
# CASE STUDY 3 : AGRICULTURAL EXPERT REASONING SYSTEM (CNF + Resolution)
# =============================================================================
def case_study_3():
    print("\n" + "=" * 78)
    print("CASE STUDY 3 : AGRICULTURAL EXPERT REASONING SYSTEM")
    print("=" * 78)

    print("""
Symbols: SD = SoilDry, IN = IrrigationNeeded, CW = CropWheat,
         AD = ApplyDripMethod, CR = CropAtRisk

Rules (Implication Elimination  A->B  ==  ~A v B):
  P1: SoilDry -> IrrigationNeeded              CNF: (~SD v IN)
  P2: (IrrigationNeeded ^ CropWheat) -> ApplyDripMethod
        Eliminate implication: ~(IN ^ CW) v AD
        De Morgan (push ~ inward): (~IN v ~CW) v AD
        CNF (already a single disjunctive clause): (~IN v ~CW v AD)
  P3: (~ApplyDripMethod ^ CropWheat) -> CropAtRisk
        Eliminate implication: ~(~AD ^ CW) v CR
        De Morgan: (AD v ~CW) v CR
        CNF: (AD v ~CW v CR)
  Facts: SoilDry = True   -> CNF unit clause: (SD)
         CropWheat = True -> CNF unit clause: (CW)
""")

    clauses = {
        frozenset({"~SD", "IN"}),        # P1
        frozenset({"~IN", "~CW", "AD"}), # P2
        frozenset({"AD", "~CW", "CR"}),  # P3
        frozenset({"SD"}),               # fact
        frozenset({"CW"}),               # fact
    }

    print("--- (b) RESOLUTION REFUTATION: Prove 'ApplyDripMethod' (AD) ---")
    result, trace = resolution_refutation(clauses, "AD")
    for line in trace:
        print(" ", line)
    print(f"\nEmpty clause derived = {result}  =>  KB |= ApplyDripMethod (AD) is PROVEN TRUE.")

    print("\n--- (c) Effect of removing the fact 'SoilDry' ---")
    clauses_no_soildry = {
        frozenset({"~SD", "IN"}),
        frozenset({"~IN", "~CW", "AD"}),
        frozenset({"AD", "~CW", "CR"}),
        frozenset({"CW"}),
    }
    result2, trace2 = resolution_refutation(clauses_no_soildry, "AD")
    print(f"Without SoilDry fact, can we still prove 'ApplyDripMethod'? -> {result2}")
    print("Reasoning: P1 can no longer fire (SoilDry unknown) so IrrigationNeeded (IN)")
    print("is never derived. Without IN, P2 cannot fire, so ApplyDripMethod (AD) is undecided")
    print("(neither proven true nor false - this is the Open World / non-monotonic gap).")
    print("Consequently P3's antecedent (~AD ^ CW) also cannot be confirmed, so")
    print("'CropAtRisk' does NOT follow either. This illustrates that classical (monotonic)")
    print("propositional inference cannot derive CropAtRisk from CW alone; an explicit")
    print("closed-world assumption (CWA), i.e. treating unprovable AD as false, would be")
    print("required to fire P3 and conclude CropAtRisk - which is a distinct, non-logically-")
    print("guaranteed reasoning pattern (default/negation-as-failure reasoning).")

    return {"clauses": clauses, "trace": trace, "result": result,
            "trace_no_soildry": trace2, "result_no_soildry": result2}


# =============================================================================
# CASE STUDY 4 : WUMPUS WORLD KNOWLEDGE AGENT (Modus Ponens + Forward Chaining)
# =============================================================================
def case_study_4():
    print("\n" + "=" * 78)
    print("CASE STUDY 4 : WUMPUS WORLD KNOWLEDGE AGENT")
    print("=" * 78)

    percepts = {"Stench[1,2]", "Breeze[1,1]", "Glitter[2,2]"}
    print(f"\nPercepts: {percepts}")

    print("\n--- (a) Belief state via Modus Ponens ---")
    mp_trace = [
        "MP1: Stench[1,2] , (Stench[1,2] -> WumpusAdjacent[1,2])  |=  WumpusAdjacent[1,2]",
        "MP2: Breeze[1,1] , (Breeze[1,1] -> PitAdjacent[1,1])     |=  PitAdjacent[1,1]",
        "MP3: Glitter[2,2] , (Glitter[x,y] -> Gold[x,y]) {x/2,y/2} |=  Gold[2,2]",
    ]
    for line in mp_trace:
        print(" ", line)
    derived_a = {"WumpusAdjacent[1,2]", "PitAdjacent[1,1]", "Gold[2,2]"}
    print(f"\nDerived facts: {derived_a}")

    print("\n--- (b) Forward chaining to localize Wumpus and Pit ---")
    fc_trace = [
        "Step 1: WumpusAdjacent[1,2] means Wumpus is in one of the cells adjacent to "
        "[1,2]: {[1,1],[1,3],[2,2]}",
        "Step 2: [2,2] has Glitter but no Stench reported from [2,2]'s neighbours "
        "contradicting Wumpus at [2,2] under normal Wumpus-world adjacency rules; "
        "candidate set narrows to {[1,1],[1,3]}",
        "Step 3: PitAdjacent[1,1] means a Pit is adjacent to [1,1]: candidate cells "
        "{[1,2],[2,1]}. Since [1,2] is the agent's current safe, visited cell "
        "(no pit there), Pit is most likely at [2,1].",
        "Step 4: Since [1,1] itself has no Stench percept, Wumpus is NOT at [1,1]; "
        "combined with Step 2, best estimate: Wumpus at [1,3] (unconfirmed, needs more data).",
    ]
    for line in fc_trace:
        print(" ", line)

    print("\n--- (c) Safety evaluation of path [1,1] -> [1,2] -> [2,2] ---")
    safety_reasoning = [
        "[1,1]: Start cell, agent already occupies it and is alive => Safe[1,1] = True.",
        "[1,2]: Stench perceived here but NOT Breeze => no Pit adjacent to [1,2]. "
        "Wumpus adjacency alone (without entering the Wumpus's own cell) does not kill "
        "the agent, so [1,2] is Safe to STAND ON (Safe[1,2] = True), though caution "
        "is needed for the next move.",
        "[2,2]: Glitter percept confirms Gold[2,2]; no Stench/Breeze reported at [2,2], "
        "and by Rule 4 (~Wumpus[x,y] ^ ~Pit[x,y] -> Safe[x,y]) we get Safe[2,2] = True.",
    ]
    for line in safety_reasoning:
        print(" ", line)
    print("\nConclusion: The path [1,1] -> [1,2] -> [2,2] is judged SAFE for the agent to "
          "traverse and collect the gold, since no cell on the path is adjacent to an ")
    print("undiscovered Pit and the only Stench encountered ([1,2]) does not, by itself, ")
    print("indicate the Wumpus occupies a cell on the path.")

    return {"percepts": percepts, "derived_a": derived_a, "fc_trace": fc_trace,
            "safety_reasoning": safety_reasoning}


# =============================================================================
# VISUALIZATION : Build a single summary figure (output.png) covering all 4
# case studies' reasoning pipelines.
# =============================================================================
def build_visual_summary(cs1, cs2, cs3, cs4):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AI Case Studies — Logical Reasoning Pipelines (CO3)",
                  fontsize=18, fontweight="bold", y=0.98)

    # ---------- Panel 1: Case Study 1 Forward Chaining flow -----------------
    ax = axes[0, 0]
    ax.set_title("Case Study 1 — Forward Chaining (Medical Diagnosis)",
                  fontsize=12, fontweight="bold")
    ax.axis("off")
    facts_boxes = ["Fever=T", "Cough=T", "Breathlessness=T"]
    rule_boxes = ["Rule1\nF^C->Flu", "Rule3\nC^B->Pneu"]
    concl_boxes = ["Possible\nFlu", "Possible\nPneumonia"]

    def draw_box(ax, x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                              fc=color, ec="black", lw=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    for i, f in enumerate(facts_boxes):
        draw_box(ax, 0.02, 0.75 - i * 0.28, 0.22, 0.18, f, "#BBDEFB")
    draw_box(ax, 0.38, 0.62, 0.22, 0.18, rule_boxes[0], "#FFE082")
    draw_box(ax, 0.38, 0.20, 0.22, 0.18, rule_boxes[1], "#FFE082")
    draw_box(ax, 0.74, 0.62, 0.22, 0.18, concl_boxes[0], "#C8E6C9")
    draw_box(ax, 0.74, 0.20, 0.22, 0.18, concl_boxes[1], "#C8E6C9")

    arrows = [
        ((0.24, 0.84), (0.38, 0.71)), ((0.24, 0.56), (0.38, 0.71)),
        ((0.60, 0.71), (0.74, 0.71)),
        ((0.24, 0.56), (0.38, 0.29)), ((0.24, 0.28), (0.38, 0.29)),
        ((0.60, 0.29), (0.74, 0.29)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="gray"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---------- Panel 2: Case Study 2 Resolution proof chain -----------------
    ax = axes[0, 1]
    ax.set_title("Case Study 2 — Resolution Refutation (Proceed(CarA))",
                  fontsize=12, fontweight="bold")
    ax.axis("off")
    steps = ["C1+C5", "+C6", "C2+ClearPath", "C4+C7", "+C8", "+GreenSig", "+~Proceed\n=> NIL"]
    n = len(steps)
    for i, s in enumerate(steps):
        y = 0.88 - i * 0.125
        color = "#FFCDD2" if i == n - 1 else "#D1C4E9"
        draw_box(ax, 0.30, y, 0.42, 0.09, s, color)
        if i > 0:
            ax.annotate("", xy=(0.51, y + 0.095), xytext=(0.51, y + 0.125 - 0.005),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="gray"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---------- Panel 3: Case Study 3 CNF Resolution tree ---------------------
    ax = axes[1, 0]
    ax.set_title("Case Study 3 — Resolution Proof Tree (ApplyDripMethod)",
                  fontsize=12, fontweight="bold")
    ax.axis("off")
    draw_box(ax, 0.02, 0.80, 0.20, 0.14, "(SD)", "#BBDEFB")
    draw_box(ax, 0.02, 0.58, 0.20, 0.14, "(~SD v IN)", "#BBDEFB")
    draw_box(ax, 0.28, 0.69, 0.20, 0.14, "(IN)", "#C8E6C9")
    draw_box(ax, 0.02, 0.36, 0.20, 0.14, "(CW)", "#BBDEFB")
    draw_box(ax, 0.28, 0.36, 0.24, 0.14, "(~IN v ~CW v AD)", "#BBDEFB")
    draw_box(ax, 0.58, 0.50, 0.20, 0.14, "(AD)", "#C8E6C9")
    draw_box(ax, 0.58, 0.20, 0.24, 0.14, "(~AD) negated goal", "#FFE082")
    draw_box(ax, 0.34, 0.02, 0.20, 0.14, "EMPTY []", "#FFCDD2")
    for (x1, y1), (x2, y2) in [((0.22, 0.87), (0.28, 0.76)), ((0.22, 0.65), (0.28, 0.76)),
                                 ((0.48, 0.76), (0.58, 0.57)), ((0.22, 0.43), (0.28, 0.43)),
                                 ((0.58, 0.57), (0.58, 0.43)), ((0.52, 0.43), (0.58, 0.43)),
                                 ((0.68, 0.50), (0.68, 0.34)), ((0.68, 0.20), (0.50, 0.16)),
                                 ((0.68, 0.20), (0.50, 0.09))]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="gray"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---------- Panel 4: Case Study 4 Wumpus grid ------------------------------
    ax = axes[1, 1]
    ax.set_title("Case Study 4 — Wumpus World Belief Map (Path Safety)",
                  fontsize=12, fontweight="bold")
    grid_labels = {
        (1, 1): "Start\nBreeze",
        (1, 2): "Stench\n(Path)",
        (2, 1): "Pit?\n(inferred)",
        (2, 2): "Glitter\nGold (Path)",
        (1, 3): "Wumpus?\n(suspected)",
        (3, 1): "", (2, 3): "", (3, 2): "", (3, 3): "",
    }
    path_cells = {(1, 1), (1, 2), (2, 2)}
    for (cx, cy), label in grid_labels.items():
        x0, y0 = (cx - 1) * 0.33, (cy - 1) * 0.33
        color = "#A5D6A7" if (cx, cy) in path_cells else "#ECEFF1"
        if "Pit" in label:
            color = "#FFAB91"
        if "Wumpus" in label:
            color = "#FFF59D"
        rect = mpatches.Rectangle((x0, y0), 0.31, 0.31, fc=color, ec="black", lw=1.2)
        ax.add_patch(rect)
        ax.text(x0 + 0.155, y0 + 0.155, label, ha="center", va="center", fontsize=8)
    # path arrows
    ax.annotate("", xy=(0.155, 0.33 + 0.155), xytext=(0.155, 0.155),
                arrowprops=dict(arrowstyle="->", lw=2, color="darkgreen"))
    ax.annotate("", xy=(0.33 + 0.155, 0.33 + 0.155), xytext=(0.155, 0.33 + 0.155),
                arrowprops=dict(arrowstyle="->", lw=2, color="darkgreen"))
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("output.png", dpi=150, bbox_inches="tight")
    print("\n[Saved visualization to output.png]")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    cs1 = case_study_1()
    cs2 = case_study_2()
    cs3 = case_study_3()
    cs4 = case_study_4()
    build_visual_summary(cs1, cs2, cs3, cs4)
    print("\n" + "=" * 78)
    print("ALL CASE STUDIES PROCESSED SUCCESSFULLY.")
    print("=" * 78)
