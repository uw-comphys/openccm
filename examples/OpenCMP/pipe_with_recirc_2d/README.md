# Instructions
1. This example contains pre-computed steady-state velocity profiles and pre-computed residence time distribution for the CFD
   tracer simulation to allow faster access to the compartmental model of the example.
   If you wish to use them, go directly to step 4.
2. Delete the `output/` directory and the `cfd_rtd.npy` file.
3. Run the `run_opencmp.py` script, this will take several minutes.
4. Run the `analysis.py` script to run CSTR- and PFR-based compartmental model and generate the comparisong figure inside `/figures`.