"""Read-only G7 constraint derivative regression probe. No calibration is run.

Usage: python docs/audits/g7_constraint_derivative_probe.py
Requires NumPy/SciPy and the repository's distortion_model package.
Prints the branch-sensitive Jacobian of active interval constraints around
the saved near-zero coefficient; the authoritative full audit remains frozen.
"""
import json
from pathlib import Path
import numpy as np
from distortion_model.joint import JointSpec
from distortion_model.centroid_bound import CentroidBound

ROOT=Path(__file__).resolve().parents[2]
G7=ROOT/"experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02"
G6=ROOT/"experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02"
model=json.loads((G7/"perturbed/model.json").read_text())
parent=json.loads((G6/"model.json").read_text())
spec=JointSpec(np.asarray(parent["b1_reference_px"]),np.asarray(parent["b4_reference_px"]),
               parent["omega1_visual_deg"],parent["omega4_visual_deg"],parent["Aref_D"])
p=spec.pack(model)
initial=spec.pack(parent)
bound=CentroidBound(spec,initial,bound_deg=1.)
print("selected p[15]",p[15],"sA_x",model["center_coefficients_reference_px"][3][0])
print("minimum constraint slack",float(np.min(bound.values(p))))
for h in (1e-5,5e-6,1e-6,5e-7,2.5e-7):
    j=bound.jacobian(p,step=h)
    active=np.flatnonzero(bound.values(p)<=1e-7)
    print("step",h,"active",active.tolist(),"active dp15",j[active,15].tolist())
print("This probe checks derivatives only; it does not certify KKT or G7.")
