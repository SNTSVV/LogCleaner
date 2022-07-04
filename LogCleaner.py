import os
import argparse
import pandas as pd
from natsort import natsorted
from src.main.operational_handler import operational_msg_removal
from src.utils.common import common_logger, convert_df_into_l_vectors

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--log', '-l', type=str, default=None, help="Input log file (including path)", required=True)
parser.add_argument('--log_size', type=float, default=None,
                    help="Log size; if less than or equal to 1 then interpreted as percentage (default: all)")
parser.add_argument('--periodicity_only', '-po', dest='periodicity_only', action='store_true',
                    help="Perform the periodicity analysis only")
parser.add_argument('--p_threshold', type=float, default=5,
                    help="Periodicity analysis threshold (default: 5, meaning 5% error is tolerable)")
parser.add_argument('--dependency_only', '-do', dest='dependency_only', action='store_true',
                    help="Perform the dependency analysis only")
parser.add_argument('--save_log', action='store_true', default=False,
                    help="Save log (file) for running LogCleaner (default: False)")
args = parser.parse_args()

# logger
logger, timestamp = common_logger('LogCleaner', level='INFO', save_log=args.save_log)

# load structured logs
logs_df = pd.read_csv(args.log)

# collect remaining templates
templates_df = logs_df[['tid', 'template']].drop_duplicates().set_index('tid')
templates_df.reindex(index=natsorted(templates_df.index))

# prepare l_vectors
l_vectors = convert_df_into_l_vectors(logs_df)

# run LogCleaner
operational_msg_removal(
    l_vectors=l_vectors,
    templates_df=templates_df,
    periodicity_only=args.periodicity_only,
    dependency_only=args.dependency_only,
    periodicity_threshold=args.p_threshold
)

logger.info('Exit LogCleaner without error(s)')
