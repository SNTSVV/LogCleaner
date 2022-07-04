import logging
import operator
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.cluster import MeanShift, estimate_bandwidth
from src.utils.common import count, ts_int, remove_tids_in_l_vectors


def operational_msg_removal(l_vectors: dict,
                            templates_df: pd.DataFrame,
                            periodicity_only: bool,
                            dependency_only: bool,
                            periodicity_threshold: float = 5.0,
                            boundary: int = 0,
                            power: int = 1,
                            min_supp: int = 2):

    logger = logging.getLogger(__name__)
    logger.info('-' * 100)
    logger.info(f'periodicity_only={periodicity_only}, '
                f'dependency_only={dependency_only}, '
                f'periodicity_threshold={periodicity_threshold}, '
                f'boundary={boundary}, '
                f'power={power}, '
                f'min_supp={min_supp}')
    logger.info('-' * 100)

    # retrieve log entries information from logs and template
    logger.info(f'Total number of executions: {len(l_vectors.keys())}')
    logger.debug(f'All execution ids: {natsorted(l_vectors.keys())}')
    logger.info(f'Total number of templates: {len(templates_df)}')
    logger.debug(f'All template ids: {natsorted(templates_df.index)}')

    counter_org = 0
    for ex, l in l_vectors.items():
        counter_org += len(l)
    logger.info(f'Total number of log entries: {counter_org}')

    if not dependency_only:
        # identify operational template ids
        periodic_tids = periodicity_analysis(l_vectors, templates_df, periodicity_threshold)

        # remove the log entries of the operational templates in l_vectors
        l_vectors = remove_tids_in_l_vectors(l_vectors, periodic_tids)
    else:
        periodic_tids = []

    if not periodicity_only:
        # add padding log entries;
        for ex_id, l_vector in l_vectors.items():
            l_vector.insert(0, {'ts': l_vector[0]['ts'], 'tid': '_init_', 'values': None})
            l_vector.append({'ts': l_vector[-1]['ts'], 'tid': '_fin_', 'values': None})

        # perform dependency_analysis
        operational_tids = distance_based_dependency_analysis(l_vectors=l_vectors, templates_df=templates_df,
                                                              boundary=boundary, power=power, min_supp=min_supp)

        # remove the log entries of the operational templates in l_vectors
        l_vectors = remove_tids_in_l_vectors(l_vectors, operational_tids+['_init_', '_fin_'])
    else:
        operational_tids = []

    counter_clean = 0
    for ex, l in l_vectors.items():
        counter_clean += len(l)
    logger.info(f'Total number of log entries after removing periodic and operational templates {counter_clean}')
    filtered_rate = (1 - (counter_clean / counter_org))
    filtered_tids = set(periodic_tids + operational_tids)
    logger.info(f'Identified Operational tids: {str(filtered_tids)}')
    logger.info(f'Filtered messages rate: {filtered_rate:.4f}')
    logger.info(f'Remaining tids: {natsorted(set(templates_df.index) - filtered_tids)}')

    return l_vectors, periodic_tids + operational_tids, filtered_rate


def periodicity_analysis(l_vectors: dict, templates_df: pd.DataFrame, p_threshold: float = 5.0):
    """
    Check the periodicity of each template and returns a set of operational templates

    :param l_vectors: logs in the form of the vectors of log entries (key: ex_id, value: l_vector)
    :param templates_df: list of templates
    :param p_threshold: threshold for periodicity error (default: 5%)
    :return: (sorted) list of operational template ids
    """
    logger = logging.getLogger(__name__)
    logger.debug('Start: periodicity_check()')

    # initialize
    periodic_tids = set()

    # decide operational for each template
    for tid in natsorted(templates_df.index):
        periodic = True  # we presume tid as periodic at first

        # check periodicity first
        for execution_id, l_vector in l_vectors.items():
            # NOTE: the periodicity check is based on each l_vector

            # get the vector of only timestamps
            ts_vector = [ts_int(entry['ts']) for entry in l_vector]
            sequence = [ts_int(entry['ts']) for entry in l_vector if entry['tid'] == tid]

            if len(sequence) == 0:  # does not occurred -> skip this log
                continue

            elif len(sequence) < 3:  # occurred less than three -> at most one timestamp diff -> not a periodic tid
                # not globally continuous, therefore not a periodic tid
                # NOTE: this can incorrectly exclude periodic events whose period is very large
                logger.info(f'tid={tid} is not globally continuous (occurred less than three times); '
                            f'ex_id={execution_id}, sequence={sequence}')
                periodic = False
                break

            # extract timestamp differences
            timestamp_diffs = list()  # timestamps' differences for all logs
            for i in range(len(sequence) - 1):
                timestamp_diffs.append(sequence[i + 1] - sequence[i])
            logger.debug(f'tid={tid}, ex_id={execution_id}, timestamp_diffs={timestamp_diffs}')

            avg_diff = np.average(timestamp_diffs)
            # check if the template occurs globally from the start until the end of the log
            if sequence[0] - ts_vector[0] <= avg_diff and ts_vector[-1] - sequence[-1] <= avg_diff:

                # make a counter for timestamp_diffs
                from collections import Counter
                counter = Counter(timestamp_diffs)

                # get the most frequently occurred time-diff value and its count
                # for example, given `timestamp_diffs=[5, 5, 2, 1, 5]`,
                # `most_common_diff=5` and `most_common_diff_counts=3`
                most_common_diff = counter.most_common(1)[0][0]
                most_common_diff_counts = counter[most_common_diff]

                # add the count for +-1 MIN_UNIT of most_common_diff, to account for a small noise in timestamps
                # for example, if the MIN_UNIT is `second`, then we consider the +-1 seconds of most_common_diff
                if most_common_diff+1 in counter.keys():
                    most_common_diff_counts += counter[most_common_diff+1]
                if most_common_diff-1 in counter.keys():
                    most_common_diff_counts += counter[most_common_diff-1]

                # compute the percentage of most_common_diff_counts
                percent = most_common_diff_counts / len(timestamp_diffs) * 100
                logger.debug(f'tid={tid} passed the globally continuous check in ex_id={execution_id}; '
                             f'percent={percent:.2f}, counter={counter}')

                # check if most_common_diff_counts is more than delta% of total
                if percent < 100 - p_threshold:
                    # not periodic
                    logger.info(f'tid={tid} is not periodic in ex_id={execution_id}, p_threshold={p_threshold}, '
                                f'percent={percent:.2f}, counter={counter}')
                    periodic = False
                    break

            else:  # not "globally" continuous, therefore not globally periodic
                logger.info(f'tid={tid} is not globally continuous in ex_id={execution_id}')
                logger.debug(f'avg_diff={avg_diff}, ts_vector[0]={ts_vector[0]}, ts_vector[-1]={ts_vector[-1]}, '
                             f'sequence={sequence}')
                periodic = False
                break

        if periodic:
            periodic_tids.add(tid)

    logger.info('-' * 100)
    logger.info('* List of periodic templates identified')
    for tid in periodic_tids:
        logger.info(f'tid={tid}, template={templates_df.loc[tid]["template"][:50]}')
    logger.info('-' * 100)
    logger.debug('End: periodicity_analysis()')

    return sorted(list(periodic_tids))


def distance_based_dependency_analysis(
        l_vectors: dict,
        templates_df: pd.DataFrame,
        boundary: int = 0,
        power: float = 1.0,
        min_supp: int = 2):

    logger = logging.getLogger(__name__)

    # get sequences which is composed of l_vector without execution_id
    tid_sequences = [[e['tid'] for e in l_vector] for execution_id, l_vector in l_vectors.items()]
    tid_sequences_rev = [[e['tid'] for e in reversed(l_vector)] for execution_id, l_vector in l_vectors.items()]
    logger.debug('tid_sequences = ')
    for tid_sequence in tid_sequences:
        logger.debug(tid_sequence)

    # get tids_with_paddings (only for y)
    tids_with_paddings = list(templates_df.index)
    tids_with_paddings.append('_init_')
    tids_with_paddings.append('_fin_')

    # for each pair of templates, calculate dScores
    dScore_forward = dict()  # collection of dScore between templates, forward
    dScore_backward = dict()  # collection of dScore between templates, backward
    for tid_x in templates_df.index:
        for tid_y in tids_with_paddings:
            if tid_x == tid_y:
                continue

            # calculate dep(x, y)
            dScore_forward[(tid_x, tid_y)] = dScore(tid_x, tid_y, tid_sequences, min_supp, power, boundary)
            # logger.debug('dScore_forward[(%s, %s)] = %.3f' %
            #               (str(tid_x), str(tid_y), dScore_forward[(tid_x, tid_y)]))
            dScore_backward[(tid_x, tid_y)] = dScore(tid_x, tid_y, tid_sequences_rev, min_supp, power, boundary > 0)
            # logger.debug('dScore_backward[(%s, %s)] = %.3f' %
            #               (str(tid_x), str(tid_y), dScore_backward[(tid_x, tid_y)]))

    # for each template, finalize the score
    mScores = dict()  # key: tid, value: dScore
    for x in templates_df.index:
        mScores[x] = 0
        for y in tids_with_paddings:
            if x == y:
                continue

            # update mScores
            if mScores[x] < dScore_forward[(x, y)]:
                mScores[x] = dScore_forward[(x, y)]
            if mScores[x] < dScore_backward[(x, y)]:
                mScores[x] = dScore_backward[(x, y)]

        # Not appearing templates (either already removed or the list of templates are not refined enough)
        if mScores[x] == 0:
            logger.debug('mScores[%s] == 0; %s' % (x, templates_df.loc[x]['template'][:70]))
            logger.debug('mScores[%s] is set to 1 to skip the template' % x)
            mScores[x] = 1

    logger.info('-' * 100)
    logger.info('List of templates whose dependency score is less then 1')
    for (tid, mScore) in sorted(mScores.items(), key=operator.itemgetter(1)):
        if mScore < 1:
            logger.info('tid=%s, mScore=%.4f, template="%s"' % (tid,
                                                                 mScore,
                                                                 templates_df.loc[tid]['template'][:50]))
    logger.info('-' * 100)

    operational_tids = cluster_scores(mScores)
    logger.info('* List of operational templates identified by the dependency analysis')
    for tid in operational_tids:
        logger.info('tid=%s, template="%s"' % (tid, templates_df.loc[tid]['template'][:50]))
    logger.info('-' * 100)

    logger.debug('End: t_based_dependency_analysis()')
    return operational_tids


def dScore(x, y, tid_sequences: list, min_supp: int, p: float, boundary: int):
    """

    :param x: index of a template (or _init_ / _fin_)
    :param y: index of another template (or _init_ / _fin_)
    :param tid_sequences: list of list of tids (without timestamp nor values in logs)
    :param min_supp: minimum number of occurrence of x (over all logs) to be considered
    :param p: (decreasing/increasing) power constant
    :param boundary: if > 0, then calculate cScore using the boundary (for experimental purpose)
    :return: average_co_occurrence_score of x followed by y (x -> y)

    NOTE
        - A log entry is composed of {'ts': str, 'tid': int, 'values': list}
    """
    # logger.debug('-' * 80)
    # logger.debug('dep(x=%d, y=%d)' % (x, y))

    # check the min_supp for all sequences as a whole
    occurrences_x = count(x, tid_sequences)
    occurrences_y = count(y, tid_sequences)
    if occurrences_x < min_supp or occurrences_y < min_supp:
        return 0

    co_occurrence_scores = list()
    for tid_sequence in tid_sequences:
        indices_x = [index for index, tid in enumerate(tid_sequence) if tid == x]
        indices_y = [index for index, tid in enumerate(tid_sequence) if tid == y]
        # logger.debug('sequences.index(sequence) = %d' % sequences.index(sequence))
        # logger.debug('indices_x = %s' % str(indices_x))
        # logger.debug('indices_y = %s' % str(indices_y))

        for index_x in indices_x:
            cScore = 0
            for i in range(index_x+1, len(tid_sequence)):
                if tid_sequence[i] == x:
                    cScore = 0
                    break
                elif tid_sequence[i] == y and i in indices_y:
                    if boundary > 0:  # for experiments
                        if i - index_x <= boundary:
                            cScore = 1
                        else:
                            cScore = 0
                    else:
                        cScore = np.true_divide(1, (i - index_x)**p)
                    # indices_y.remove(i)  # to avoid redundant uses of the same log entry
                    break
            co_occurrence_scores.append(cScore)
            # logger.debug('index_x = %3d\tcScore = %.3f' % (index_x, cScore))

    # sum up all the co_occurrence_scores and divide by total occurrences of x
    assert occurrences_x == len(co_occurrence_scores)
    average_co_occurrence_score = np.average(co_occurrence_scores)
    # logger.debug('dep(x=%d, y=%d)\taverage_co_occurrence_score = %.3f' % (x, y, average_co_occurrence_score))
    return average_co_occurrence_score


def cluster_scores(susp_scores: dict):
    """
    Clustering-based segmentation of templates in terms of their scores

    :param susp_scores: [key=tid, value=dependency_score]
    :return: list of operational tids
    """
    logger = logging.getLogger(__name__)

    scores = sorted(susp_scores.items(), key=operator.itemgetter(1))  # list of (tid, score)

    # Perform 1D clustering using the Mean-Shift algorithm
    X = np.array([score[1] for score in scores]).reshape(-1, 1)
    estimated_bandwidth = estimate_bandwidth(X)
    if estimated_bandwidth < 0.01:
        logger.info('estimated_bandwidth=%f => adjust to 0.1' % estimated_bandwidth)
        estimated_bandwidth = 0.01
    logger.info('bandwidth=%.3f' % estimated_bandwidth)
    cluster_labels = MeanShift(bandwidth=estimated_bandwidth).fit_predict(X)
    logger.info('cluster_labels = %s' % str(cluster_labels))

    # Find the segmentation index using the clustering labels
    segmentation_index = 0
    for i in range(len(scores)-1):
        if cluster_labels[i] != cluster_labels[i+1]:
            segmentation_index = i + 1
            break

    return [scores[i][0] for i in range(0, segmentation_index)]
