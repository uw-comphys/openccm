# Converting 
TODO: How much do we expect them to have to do themselves?

# Obtaining OpenFOAM simulation results
1. Run `foamRun` inside the `velocity_profile` directory.
2. Copy the U, p, phi, and Vc from the final timestep in `velocity_profile` into `scalar_transport/0`.
3. Inside `scalar_transport/system/controlDict`, make sure `endTime` is set to `20` and both deltaT and writeInterval are set to `0.01`.
4. Run `foamRun` inside `scalar_transport`.
5. Copy U, p, phi, and Vc from `scalar_transport/0` into `scalar_transport/20`.
6. Change `endTime` to `100` and both deltaT and writeInterval are set to `0.5`.
7. Run `foamRun` inside `scalar_transport`.
8. Copy U, p, phi, and Vc from `scalar_transport/20` into `scalar_transport/100`.
9. Change `endTime` to `300` and both deltaT and writeInterval are set to `1`.
10. Run `foamRun` inside `scalar_transport`.

# Running OpenCCM results and analysis 
1. Run the `analyze_results.py` file, it will run OpenCCM and calculate and compare the RTD