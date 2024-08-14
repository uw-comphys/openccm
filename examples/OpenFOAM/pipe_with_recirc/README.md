# Instructions
1. This example requires OpenFOAM 12, make sure it's installed.
2. If you wish to change mesh density, open the .geo file in gmsh, make your desired changes, and export as a .msh
3. Convert to OpenFOAM format using `gmshToFoam`
4. Open the `boundary` file inside the `constant` folder and change the type for each of the physical surfaces as follows:
For running this simulation in your system you have to follow these steps :
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
4. Copy the `_0` folder and rename it `0`
5. Run the simulation by calling `foamRun`.
6. Delete all of the timestep folders except for the final one.
7. Copy `0/T` into the final timestep folder.
8. Rename the final timestep folder to `0`.
9. Open `controlDict`, comment out the solve line with `incompressibleFluid` and uncomment the two lines below it defining a new solve and subSulver, finally uncomment the `#includeFunc scalarTransport` line at the bottom of the file.
10. Inside `controlDict` change endTime to 300, deltaT to 0.2, and writeInterval to 0.2.
11. Inside `fvSchemes` change `ddtSchemes` from `steadyState` to `Euler`. 
12. Run with `foamRun`.
13. Run the `analysis.py` script