"""Module containing functions to compute velocities."""
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas


from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    SpeedCalculation,compute_individual_speed
)

def compute_pair_distibution_function(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    radius_max: float,
    dx:float,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    speed_calculation: SpeedCalculation = SpeedCalculation.BORDER_EXCLUDE,
) :
    
    speed = compute_individual_speed(
    traj_data=traj_data,
    frame_step=frame_step,
    compute_velocity=True,
)

    df = pandas.merge(traj_data, speed, on=['id', 'frame'])[['id', 'frame', 'x', 'y', 'v_x', 'v_y']]
    # Create DataFrame with all mutual distances, TTCs, and Rates of Approach
    df_pd = calculate_data_frame_pair_dist(df)

    # Scramble time-information to mitigate finite-size effects and calculate mutual distances of scrambled dataset
    df.Frame = df.Frame.sample(frac=1).reset_index(drop=True)
    df_pd_ni = calculate_data_frame_pair_dist(df)

    radius_bins = np.arange(0, radius_max,dx)

    # Calculate pair distibution: g(r)
    pair_distibution =calc_pair_dist(df_pd.r, df_pd_ni.r, radius_bins, dx)
    
    return radius_bins, pair_distibution

# Function to calculate DataFrame of pairwise distances
def calculate_data_frame_pair_dist(df):
    gdf = df.groupby('Frame')

    #Compute the number of possible combinaison of ID per frame
    N_val = 0  
    for _, df_f in gdf:
        ids = len(df_f['ID'].unique())  # Number of unique IDs
    N_val += int(ids * (ids - 1) / 2)  # Calculate combinations of IDs


    observables = np.empty((N_val, 3))  # Initialize array to store results
    index = 0
    for _, df_f in gdf:  # Iterate over groups (frames)
        gdf_f_id = df_f.groupby('ID')
        for i, (_, df_i) in enumerate(gdf_f_id):
            for _, df_j in list(gdf_f_id)[i+1:]:
                init_observables(observables, df_i, df_j, index)  # Populate array with pairwise distances
                index += 1
    return pandas.DataFrame({'r': observables[:, 0], 'ttc': observables[:, 1], 'roa': observables[:, 2]})

# Function to initialize observables for a pair of data frames
def init_observables(observables, df_i, df_j, index):
    observables[index, 0] = d(df_i['x'].iloc[0], df_j['x'].iloc[0])  # Calculate distance
    observables[index, 1] = ttc(df_i['x'].iloc[0], df_j['x'].iloc[0], df_i['v_x'].iloc[0], df_j['v_x'].iloc[0])  # Calculate time-to-collision
    observables[index, 2] = rate_of_approach(df_i['x'].iloc[0], df_j['x'].iloc[0], df_i['v_x'].iloc[0], df_j['v_x'].iloc[0])  # Calculate rate of approach

# Function to calculate time-to-collision
def ttc(x1, x2, v1, v2, l_a=0.2, l_b=0.2):
    if (v1-v2) !=0:
        ttc = ((x2-x1)-(0.5*(l_a+l_b)))/(v1-v2)
        if ttc>0:
            return ttc  # Formula for time-to-collision
        else:
            return 99999.9  # Return a large value if conditions are not met
    else:
        return 99999.9  # Return a large value if conditions are not met

# Function to calculate absolute difference
def d(a, b):
    return abs(a - b)

# Function to calculate rate of approach
def rate_of_approach(x_a, x_b, v_a, v_b):
    return -1 * (v_a - v_b) * (x_a - x_b) / d(x_a, x_b)

# Function to calculate pairwise distances
def calc_pair_dist(x, x_scrambled, x_bins, dx):
    x1=calc_dist(x, x_bins, dx)
    x2=calc_dist(x_scrambled, x_bins, dx)
    # x2 = np.ones((len(x1)))*(1/len(x1)) # Equi-distribution
    ### division by 0 typicaly 0/0 are set to == 0
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     c = np.divide(x1,x2)
    #     c[c == np.nan] = 0
    #     c = np.nan_to_num(c)
    c =x1/x2
    return c
    


# Function to calculate distances
def calc_dist(x, x_bins, dx):
    x_min = min(x_bins)
    x_max = max(x_bins)
    p_x = np.zeros(len(x_bins))  # Initialize array to store counts
    for x_i in x:
        if x_min < x_i < x_max+dx:
            p_x[int(np.floor((x_i-x_min)/dx))] += 1  # Increment count for corresponding bin
    return p_x / len(x)  # Calculate probability distribution
