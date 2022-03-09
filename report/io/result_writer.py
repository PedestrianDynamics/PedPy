import pathlib
from typing import Dict

from report.methods.method_a import ResultMethodA


def write_results(
    output_directory: pathlib.Path,
    traj_file_name: str,
    frame_rate: float,
    results_method_a: Dict[int, ResultMethodA],
):
    # create output directory
    output_directory.mkdir(exist_ok=True)

    # write all results
    write_method_a_results(results_method_a, output_directory, traj_file_name, frame_rate)


def write_method_a_results(
    results: Dict[int, ResultMethodA],
    output_directory: pathlib.Path,
    traj_file_name: str,
    frame_rate: float,
):
    method_a_output_directory = output_directory / "Fundamental_Diagram" / "FlowVelocity"
    method_a_output_directory.mkdir(parents=True, exist_ok=True)

    for line_id, result in results.items():
        with open(
            method_a_output_directory / f"Flow_NT_{traj_file_name}_id_{line_id}.dat", "w"
        ) as nt_output_file:
            nt_output_file.write(
                (f"#framerate:	{frame_rate}\n" "\n" "#Frame	Time [s]	Cumulative pedestrians\n")
            )

            result.df_nt.to_csv(
                nt_output_file,
                sep="\t",
                header=False,
                mode="a",
                index_label=False,
                index=False,
                float_format="%.2f",
            )

        with open(
            method_a_output_directory / f"FDFlowVelocity_{traj_file_name}_id_{line_id}.dat", "w"
        ) as flow_output_file:
            flow_output_file.write("#Flow rate(1/s)     Mean velocity(m/s)\n")
            result.df_flow.to_csv(
                flow_output_file,
                sep="\t",
                header=False,
                mode="a",
                index_label=False,
                index=False,
                float_format="%.2f",
            )
