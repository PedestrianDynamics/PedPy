import pandas as pd

from pedpy.column_identifier import *
from pedpy.io.helper import TrajectoryUnit


def prepare_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data set for comparison with the result of the parsing.

    Trims the data frame to the first 4 columns and sets the column dtype to
    float. This needs to be done, as only the relevant data are read from the
    file, and the rest will be ignored.

    Args:
        data_frame (pd.DataFrame): data frame to prepare

    Result:
        prepared data frame
    """
    result = data_frame.iloc[:, :4].copy(deep=True)
    result = result.astype("float64")

    return result


def get_data_frame_to_write(data_frame: pd.DataFrame, unit: TrajectoryUnit) -> pd.DataFrame:
    """Get the data frame which should be written to file.

    Args:
        data_frame (pd.DataFrame): data frame to prepare
        unit (TrajectoryUnit): unit used to write the data frame

    Result:
        copy
    """
    data_frame_to_write = data_frame.copy(deep=True)
    if unit == TrajectoryUnit.CENTIMETER:
        data_frame_to_write[2] = pd.to_numeric(data_frame_to_write[2]).mul(100)
        data_frame_to_write[3] = pd.to_numeric(data_frame_to_write[3]).mul(100)
    return data_frame_to_write
