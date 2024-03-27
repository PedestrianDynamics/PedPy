import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import pedpy


# Set path to data directory
path = "./notebooks/demo-data/single_file/"

# Read trajectory files
# files = os.listdir(path)

# Choose data file
file = "n56_cam1.csv"

### Read data and create DataFrame
df = pd.read_csv(os.path.join(path, file), comment="#")
df = df.rename(columns={'ID': 'id', 'Frame': 'frame'})[["id","frame","x","y"]]
df["frame"]
traj = pedpy.TrajectoryData(df,25)

# traj=pedpy.load_trajectory_from_ped_data_archive_hdf5(pathlib.Path(path+"00_01a.h5"))




radius_bins, pair_distibution = pedpy.compute_pair_distibution_function(
    traj_data=traj, radius_bin_size=0.1
)

fig, (ax1) = plt.subplots(1, figsize=(5, 5))

# Plot g(r) on the first subplot
ax1.plot(radius_bins, pair_distibution)
plt.show()
