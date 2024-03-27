"""Module containing functions to compute velocities."""
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas

from pedpy.column_identifier import  FRAME_COL, ID_COL,X_COL,Y_COL,V_X_COL,V_Y_COL,POINT_COL
from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.speed_calculator import (
    SpeedCalculation,
    compute_individual_speed)


def compute_pair_distibution_function(
    *,
    traj_data: TrajectoryData,
    radius_bin_size: float
):

    df =traj_data.data

    # Create Dataframe with all mutual distances, TTCs, and Rates of Approach
    dist_pd_array = calculate_data_frame_pair_dist(df)

    # Scramble time-information to mitigate finite-size effects and calculate mutual distances of scrambled dataset
    df.frame = df.frame.sample(frac=1).reset_index(drop=True)
    dist_pd_ni_array = calculate_data_frame_pair_dist(df)

    ## Create the bin for data
    radius_bins = np.arange(0, dist_pd_array.max(), radius_bin_size)

    # Calculate pair distibution: g(r)
    Nb_dist=len(dist_pd_array)
    ## Actual distribution
    pd_bins=pandas.cut(dist_pd_array,radius_bins)
    pd_bins_normalised = (pd_bins.value_counts().sort_index().to_numpy())/Nb_dist
    ## Scrambled distribution
    pd_ni_bins=pandas.cut(dist_pd_ni_array,radius_bins)
    pd_ni_bins_normalised = (pd_ni_bins.value_counts().sort_index().to_numpy())/Nb_dist

    pair_distibution = pd_bins_normalised/pd_ni_bins_normalised

    return radius_bins[1:], pair_distibution


# Function to calculate Dataframe of pairwise distances
def calculate_data_frame_pair_dist(df):
    gdf = df.groupby(FRAME_COL)

    # Compute the number of possible combinaison of id per frame
    N_val = 0
    for _, df_f in gdf:
        ids = len(df_f[ID_COL].unique())  # Number of unique ids
        N_val += int(ids * (ids - 1) / 2)  # Calculate combinations of ids

    dist_array = np.empty((N_val))  # Initialize array to store results
    index = 0 
    for _, df_f in gdf:  # Iterate over groups (frames)
        gdf_f_id = df_f.groupby(ID_COL)
        for i, (_, df_i) in enumerate(gdf_f_id):
            for _, df_j in list(gdf_f_id)[i + 1 :]:
                # Populate array with pairwise distances
                dist_array[index] =(df_j[POINT_COL].iloc[0]).distance(df_i[POINT_COL].iloc[0])
                index += 1
    return dist_array





# # Function to initialize observables for a pair of data frames
# def init_observables(observables, df_i, df_j, index):

#     observables[index, 1] = ttc(
#         df_i[X_COL].iloc[0],
#         df_j[X_COL].iloc[0],
#         df_i[V_X_COL].iloc[0],
#         df_j[V_X_COL].iloc[0],
#     )  # Calculate time-to-collision
#     observables[index, 2] = rate_of_approach(
#         df_i[X_COL].iloc[0],
#         df_j[X_COL].iloc[0],
#         df_i[V_X_COL].iloc[0],
#         df_j[V_X_COL].iloc[0],
#     )  # Calculate rate of approach


## Function to calculate time-to-collision
# def ttc(x1, x2, v1, v2, l_a=0.2, l_b=0.2):
#     if (v1 - v2) != 0:
#         ttc = ((x2 - x1) - (0.5 * (l_a + l_b))) / (v1 - v2)
#         if ttc > 0:
#             return ttc  # Formula for time-to-collision
#         else:
#             return 99999.9  # Return a large value if conditions are not met
#     else:
#         return 99999.9  # Return a large value if conditions are not met



# # Function to calculate rate of approach
# def rate_of_approach(x_a, x_b, v_a, v_b):
#     return -1 * (v_a - v_b) * (x_a - x_b) / d(x_a, x_b)



