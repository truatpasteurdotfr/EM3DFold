"""Small OR-Tools helpers shared by assembly-style pipelines."""

from __future__ import annotations

from ortools.sat.python import cp_model


def solve(score1d, rst2d, hint=None, time_limit=1200, num_workers=2, log_search_progress=True, **kwargs):
    del kwargs
    model = cp_model.CpModel()

    print("# Preprocess 1/3 add decision variable")
    n = len(score1d)
    x = [model.NewBoolVar(f"x[{i}]") for i in range(n)]

    for i in range(n):
        rst2d[i][i] = 1

    print("# Preprocess 2/3 add rst")
    for i in range(n):
        model.Add(sum(x[k] for k in range(n) if rst2d[i][k] == 1) <= 1).OnlyEnforceIf(x[i])

    if hint is not None:
        for idx in hint:
            model.AddHint(x[idx], True)

    print("# Preprocess 3/3 set objectives")
    objectives = [score1d[i] * x[i] for i in range(n)]
    model.Maximize(sum(objectives))

    print("# Solving")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = log_search_progress
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)

    results = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if solver.BooleanValue(x[i]):
                results.append(i)
    return results, solver.ObjectiveValue(), status
