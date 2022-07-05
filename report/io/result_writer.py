import csv
import pathlib
from typing import Dict

from report.methods.method_CCM import ResultMethodCCM


def write_results(
    output_directory: pathlib.Path,
    traj_file_name: str,
    frame_rate: float,
    results_method_ccm: Dict[int, ResultMethodCCM],
):
    # create output directory
    output_directory.mkdir(exist_ok=True)

    # write all results
    write_method_ccm_results(results_method_ccm, output_directory, traj_file_name, frame_rate)


def write_method_ccm_results(
    results: Dict[int, ResultMethodCCM],
    output_directory: pathlib.Path,
    traj_file_name: str,
    frame_rate: float,
):
    method_ccm_output_directory = output_directory / "Fundamental_Diagram" / "CCM_Voronoi"
    method_ccm_output_directory.mkdir(parents=True, exist_ok=True)

    for line_id, result in results.items():
        with open(
            method_ccm_output_directory / f"rho_v_{traj_file_name}_id_{line_id}.dat", "w"
        ) as mean_output_file:
            mean_output_file.write(f"#framerate:	{frame_rate}\n\n")
            result.df_mean.to_csv(
                mean_output_file,
                mode="a",
                sep="\t",
                header=True,
                index_label=False,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )

        with open(
            method_ccm_output_directory / f"rho_v_{traj_file_name}_direction_id_{line_id}.dat", "w"
        ) as mean_output_file:
            mean_output_file.write(f"#framerate:	{frame_rate}\n\n")
            result.df_direction.to_csv(
                mean_output_file,
                mode="a",
                sep="\t",
                header=True,
                index_label=False,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )

        with open(
            method_ccm_output_directory / f"ICCM_{traj_file_name}_id_{line_id}.dat", "w"
        ) as individual_output_file:
            individual_output_file.write(f"#framerate:	{frame_rate}\n\n")

            result.df_individual.to_csv(
                individual_output_file,
                mode="a",
                sep="\t",
                header=True,
                index_label=False,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
