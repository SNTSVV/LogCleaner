import os
import re
import sys
import random
import calendar
import pandas as pd
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

abbr_to_num = {v: k for k, v in enumerate(calendar.month_abbr)}


def remove_tids_in_l_vectors(l_vectors: dict, tids: list):
    if len(tids) == 0:
        return l_vectors

    updated_l_vectors = dict()
    for execution_id, l_vector in l_vectors.items():
        updated_l_vector = [entry for entry in l_vector if entry['tid'] not in tids]
        if len(updated_l_vector) > 0:
            updated_l_vectors[execution_id] = updated_l_vector

    return updated_l_vectors


def ts_int(timestamp):
    try:
        if type(timestamp) == int or type(timestamp) == float:
            return timestamp

        patterns = [r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})[,|.](\d{3})',
                    r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}).(\d{3})Z',
                    r'(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})',
                    r'(\w+) (\d+) (\d{2}):(\d{2}):(\d{2})']
        for pattern in patterns:
            m = re.match(pattern, timestamp)
            if m:
                y, month, d, h, minute, s, ms = 0, 0, 0, 0, 0, 0, 0
                if len(m.groups()) == 7:  # `2015-10-17 15:37:56,547`
                    y, month, d, h, minute, s, ms = m.groups()
                elif len(m.groups()) == 6:  # '16/04/07 10:46:05'
                    y, month, d, h, minute, s = m.groups()
                elif len(m.groups()) == 5:  # 'Jun 9 00:01:10'
                    month, d, h, minute, s = m.groups()
                    month = abbr_to_num[month]
                else:
                    print(f'ERROR: no care for this case, timestamp={timestamp}')
                    exit(-1)
                return int(s) + int(minute) * 60 + int(h) * 60 * 60 + int(d) * 60 * 60 * 24 \
                    + int(month) * 60 * 60 * 24 * 31 + int(y) * 60 * 60 * 24 * 31 * 12

        if len(timestamp) == 13:
            # SES logs
            # ex: timestamp = '2019127100455'
            y = timestamp[0:4]
            d = timestamp[4:7]
            h = timestamp[7:9]
            m = timestamp[9:11]
            s = timestamp[11:13]

            return (int(d) * 60 * 60 * 24) + (int(h) * 60 * 60) + (int(m) * 60) + int(s)

        elif len(timestamp) == 6:
            # Proxifier
            # ex: timestamp = '164909'
            h = timestamp[0:2]
            m = timestamp[2:4]
            s = timestamp[4:6]

            return (int(h) * 60 * 60) + (int(m) * 60) + int(s)

        elif re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp):
            # 2015-10-17 15:37:56,547
            m = re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp)
        else:
            return int(timestamp)
    except ValueError as e:
        print(f'Cannot handle timestamp={timestamp} - {e}')
        exit(-1)


def count(tid: str, tid_sequences: list):
    counter = 0
    for tid_sequence in tid_sequences:
        counter += tid_sequence.count(tid)
    return counter


def common_logger(name: str, level='DEBUG', save_log=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_id = timestamp + f'_{os.getpid()}'

    if save_log:
        if not os.path.exists('_logs'):
            os.makedirs(os.path.join('_logs'))
        log_file = os.path.join('_logs', f'{name}_{log_id}.log')
        logging.basicConfig(filename=log_file,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(stream=sys.stdout,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    lg = logging.getLogger()
    if level == 'DEBUG':
        lg.setLevel(logging.DEBUG)
    elif level == 'INFO':
        lg.setLevel(logging.INFO)
    return lg, log_id


def convert_df_into_l_vectors(logs_df: pd.DataFrame, num_logs=None, include_component=False):
    """
    Convert the format of logs from pandas dataframe into l_vectors (dict, key: log_id, value: a log).

    :param logs_df: logs (pandas dataframe)
    :param num_logs: (optional) the number of logs want to convert / for experiments
    :param include_component: (optional, default=False) True to include component information in l_vectors
    :return: l_vectors (dict, key: log_id, value: a log = a list of log entries)
    """
    logs_df = logs_df.copy()
    header = set(logs_df.columns)
    if {'month', 'date', 'time'}.issubset(header):
        # (ex) month = Jun, date = 9, time = 06:06:20
        logs_df['ts'] = logs_df.apply(lambda x: ' '.join([str(x['month']), str(x['date']), str(x['time'])]), axis=1)
    elif {'date', 'time'}.issubset(header):
        # (ex) date = 2015-10-17, time = 15:37:56,547
        # (ex) date = 16/04/07, time=10:46:05
        logs_df['ts'] = logs_df.apply(lambda x: ' '.join([str(x['date']), str(x['time'])]), axis=1)
    elif {'time'}.issubset(header):
        # (ex) time = 2020-03-08T23:01:10.016Z
        logs_df = logs_df.rename(columns={'time': 'ts'})
    else:
        print(f'WARNING: No timestamp in the logs:\n{logs_df.head()}')
        logs_df['ts'] = logs_df['lineID']

    l_vectors = dict()
    reduced_header = ['ts', 'tid', 'values']
    if include_component:
        reduced_header.append('component')
    # logs_df = logs_df[['logID'] + reduced_header]
    if 'logID' in logs_df.columns:
        for log_id in logs_df['logID'].unique():
            log_df = logs_df[logs_df['logID'] == log_id][reduced_header]
            l_vector = log_df.to_dict('records')
            l_vectors[log_id] = l_vector
    else:
        log_df = logs_df[reduced_header]
        l_vector = log_df.to_dict('records')
        l_vectors[1] = l_vector

    # use subset of all logs if specified
    if num_logs:
        print(f'Use only {num_logs} logs among {len(l_vectors.keys())} logs')
        logger.info(f'Use only {num_logs} logs among {len(l_vectors.keys())} logs')
        log_ids = random.sample(list(l_vectors.keys()), k=len(l_vectors.keys()) - num_logs)
        for log_id in log_ids:
            del l_vectors[log_id]
    else:
        logger.info(f'Total number of logs: {len(l_vectors.keys())}')

    return l_vectors
