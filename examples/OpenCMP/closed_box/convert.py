from opencmp.config_functions import ConfigParser
from opencmp.post_processing import sol_to_vtu_direct

# config = ConfigParser("config_Stokes")
config = ConfigParser("config_INS")
# config = ConfigParser("config_MCINS")
out_dir_path = "output/"

if __name__ == "__main__":
    sol_to_vtu_direct(config, out_dir_path)
    print("Done")
