# Create a fake OpenCMP result to be converted into a single CSTR/PFR
import os
from pathlib import Path

import ngsolve as ngs
from netgen.read_gmsh import ReadGmsh

Path('output/ins_sol/').mkdir(parents=True, exist_ok=True)

# Define either the velocity or the space time, depending on what's more natural for
# your particular case
# velocity = 1
# tau = 1.0 / velocity
# print(f"Space time is {tau}.")
tau = 10
velocity = 1.0 / tau
print(f"Velocity is {velocity}")

mesh = ngs.Mesh(ReadGmsh("v=1.msh"))

fes_u = ngs.VectorH1(mesh, order=3)
fes_p = ngs.H1(mesh, order=2)
_fes = [fes_u, fes_p]
fes = ngs.FESpace(_fes)

gfu = ngs.GridFunction(fes)
gfu.components[0].Interpolate(ngs.CoefficientFunction((velocity, 0)))
gfu.Save('output/ins_sol/ins_0.0.sol')
ngs.VTKOutput(ma=mesh, coefs=[c for c in gfu.components], names=['velocity', 'fake pressure'],
              filename='output/ins_sol/ins_0.0', subdivision=1).Do()
