import argparse
import sys
import textwrap
import time

from report.io.geometry_parser import parse_geometry
from report.io.ini_parser import parse_ini_file
from report.io.result_writer import write_results
from report.io.trajectory_parser import parse_trajectory
from report.methods.method_CCM import run_method_ccm
from report.util.loghelper import *
from report.version import __version__


def main():
    sys.exit(Application().run())


class Application:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(
                """\
                      _ ____  ____                            _   
                     | |  _ \/ ___| _ __ ___ _ __   ___  _ __| |_ 
                  _  | | |_) \___ \| '__/ _ \ '_ \ / _ \| '__| __|
                 | |_| |  __/ ___) | | |  __/ |_) | (_) | |  | |_ 
                  \___/|_|   |____/|_|  \___| .__/ \___/|_|   \__|
                                            |_|                   
                -----------------------------
                JPSreport is a command line module to analyze trajectories of pedestrians.
                """
            ),
        )
        self.setup_logging()
        self.setup_arg_parser()
        self.args = None

    def setup_logging(self):
        logging.basicConfig(format=logging.BASIC_FORMAT)

    def upgrade_logging(self):
        logger_name = "JPSreport"
        if self.args.v >= 1:
            logging.getLogger(logger_name).setLevel(logging.INFO)
        if self.args.v >= 2:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
        if self.args.v >= 3:
            # Enables all log messages from 3rd party libraries
            logging.getLogger().setLevel(logging.DEBUG)

    def setup_arg_parser(self):
        self.parser.add_argument(
            "-v",
            action="count",
            default=1,
            help="Set verbosity level, use -v for info messages, "
            "-vv for debug and -vvv for everything",
        )
        self.parser.add_argument(
            "ini_file",
            default="ini-file.xml",
            help="ini-file containing the configuration of the analysis to run",
            type=argparse.FileType("r", encoding="UTF-8"),
        )
        self.parser.add_argument(
            "--version", action="version", version="%(prog)s {version}".format(version=__version__)
        )

    def parse_arguments(self):
        self.args = self.parser.parse_args()

    def run(self):
        self.parse_arguments()
        self.upgrade_logging()
        self.run_analysis()

    def run_analysis(self):
        log_info("Starting JuPedSim - JPSreport")
        start_time = time.time()

        configuration = parse_ini_file(self.args.ini_file)
        geometry = parse_geometry(configuration.geometry_file)

        for trajectory_file in configuration.trajectory_files:
            log_info(f"Start Analysis for the file: {trajectory_file.name}")

            trajectory_data = parse_trajectory(trajectory_file)

            results_method_ccm = run_method_ccm(
                configuration.config_method_ccm,
                trajectory_data,
                configuration.measurement_lines,
                geometry,
                configuration.velocity_configuration,
            )

            write_results(
                configuration.output_directory,
                trajectory_file.name,
                trajectory_data.frame_rate,
                results_method_ccm,
            )

            log_info(f"End Analysis for the file: {trajectory_file.name}")

        end_time = time.time()

        log_info("Finishing...")
        log_info(f"Time elapsed:\t {end_time - start_time:.2f} s\n")
