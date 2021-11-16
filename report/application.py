import argparse
import textwrap
from typing import Final

from report.util.loghelper import *


class Application:
    JPS_REPORT_VERSION: Final = "1.0.0"

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
            "ini-file.xml",
            help="ini-file containing the configuration of the analysis to run",
            type=argparse.FileType("r", encoding="UTF-8"),
        )

    def parse_arguments(self):
        self.args = self.parser.parse_args()

    def run(self):
        self.parse_arguments()
        self.upgrade_logging()
