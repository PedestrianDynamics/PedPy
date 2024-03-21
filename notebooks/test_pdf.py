import os
import pandas as pd
import pedpy
import matplotlib.pyplot as plt
import pathlib

# Set path to data directory
path = "./notebooks/demo-data/single_file/"

# Read trajectory files
files = os.listdir(path)

# Choose data file
file = "n34_cam2.csv"

# Read data and create DataFrame
df = pd.read_csv(os.path.join(path, file), comment="#")



traj = pedpy.load_trajectory(
    trajectory_file=pathlib.Path("./notebooks/demo-data/bottleneck/040_c_56_h-.txt")
)

radius_bins, pair_distibution = pedpy.compute_pair_distibution_function(
    traj_data=traj,
    frame_step=25,
    radius_max=3,
    dx=0.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot g(r) on the first subplot
ax1.plot(radius_bins, pair_distibution)
