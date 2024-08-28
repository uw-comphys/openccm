# Instructions
1. This example requires OpenFOAM 12, make sure it's installed.
2. The example includes the steady state velocity profile inside `0/`.
   If you wish to use it, go directly to step 13.
   Otherwise, if you wish to change the mesh resolution or any simulation parameters, keep reading the steps in order.
3. Convert to OpenFOAM format using `gmshToFoam` command in the terminal.
4. Open the `boundary` file inside the `constant` folder and change the type for each of the physical surfaces as follows:
For running this simulation in your system you have to follow these steps:
    top_and_bottom
    {
        type            empty;
    }
    wall
    {
        type            wall;
    }
    inlet
    {
        type            patch;
    }
    outlet
    {
        type            patch;
    }
5. Delete `0/`, copy `_0/` and rename it `0`
6. Open `system/fvSchemes` and change the ddtSchemes from `Euler` to `steadyState`. 
7. Open `system/controlDict`.
   Comment out the `#includeFunc scalarTransport` line. 
   Uncomment `solver          incompressibleFluid;`.
   Comment out the two lines below `// For tracer`.
   Change `endTime` to 2000, `deltaT` to 0.2, and `writeInterval` to 0.2. 
8. Run the simulation by calling `foamRun` in the terminal. 
9. Delete all of the timestep folders except for the final one. 
10. Copy `_0/T` into the final timestep folder. 
11. Rename the final timestep folder to `0`. 
12. Undo the changes made to `fvSchemes` and `controlDict`. 
13. Run using the `foamRun` command in the terminal. 
14. Run the `analysis.py` script.