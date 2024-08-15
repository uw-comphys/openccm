from opencmp.run import run

if __name__ == "__main__":
    # Run Stokes
    run("config_Stokes")

    # Run INS
    run("config_INS")

    # Run MCINS
    run("config_MCINS")
