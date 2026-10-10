# G7 attempt 02 resume pointer

**COMPLETE_UNCERTIFIED / PAUSE.** The declared 32-outer, two-start constrained campaign finished. No continuation is authorized; G8 has not run.

Selected snapshot: [perturbed solution](perturbed/solution.npz), [summary](summary.json), [checkpoint](checkpoint.json). The compatible G6 initializer remains the last usable certified checkpoint: [G6 attempt 02](../g6_attempt_02/checkpoint.json).

The user-authorized one-degree bound is feasible over the full interval domain, and independent NumPy objective/forward reconstruction matches the saved result. Numerical certification fails: selected global KKT residual is 2.8122e−4 versus 1e−6; active-constraint curvature finite-difference agreement is 0.4508 versus 0.01, so constrained profile curvature was not evaluated. These are separate certificate limitations. Individual compact inferences are progress diagnostics only.

One next numerical question: why does the constrained interval formulation have unstable active-constraint Hessian near the boundary, and can a smooth equivalent certificate preserve the same physical bound and objective? Do not expand the model or launch another fit before resolving that question.
