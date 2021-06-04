import re
import tempfile
import subprocess
import operator
import collections
import logging

logger = logging.getLogger(__name__)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k,v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k,v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["reference-coreference-scorers/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=True):
    with tempfile.NamedTemporaryFile(delete=True, mode="w") as prediction_file:
        with open(gold_path, "r") as gold_file:
            output_conll(gold_file, prediction_file, predictions, subtoken_maps)
        # logger.info("Predicted conll file: {}".format(prediction_file.name))
        results = {m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
    return results

# if __name__ == "__main__":
#     import sys
#     psum, rsum, f1sum = 0,0,0
#     for m in ("muc", "bcub", "ceafe"):
#         res = official_conll_eval(sys.argv[1], sys.argv[2], m)
#         print('metric %s = %s' % (m, str(res)))
#         rsum += res['r']
#         psum += res['p']
#     p = psum/3.0
#     r = rsum/3.0
#     f1 = 2*p*r/(p+r)
#     print('%s\t%s\t%s' % (p,r,f1))
#

if __name__ == "__main__":
    # methods = ['exact', 'sym', 'nystrom', 'cur', 'cur_alt', 'nystrom_eig_est', 'sms_nystrom', 'nystrom_eig_est_rescale',
    #            'sms_nystrom_rescale']
    methods = ['exact', 'sym', 'nystrom', 'cur', 'cur_alt', 'nystrom_eig_est_rescale',
               'sms_nystrom_rescale', 'skeleton']
    range_of_approx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    import copy
    import os
    goldfile = 'data/ecb/gold/test_mixed_topic_level.conll'
    base_pred = 'models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40/low_rank/%s/%s/test_mixed_average_0.65_model_3_topic_level.conll'
    import pickle
    error_dict = pickle.load(open('models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40/low_rank/results_low_rank.pkl', 'rb'))[0]
    print('res_f1 = {')
    for m in methods:
        x_axis = []
        f1s = []
        errors = []
        for rank in range_of_approx:
            predfile = copy.copy(base_pred) % (m, rank)
            if os.path.exists(predfile):
                psum, rsum, f1sum = 0, 0, 0
                for metric in ("muc", "bcub", "ceafe"):
                    res = official_conll_eval(goldfile, predfile, metric)
                    # print('metric %s = %s' % (metric, str(res)))
                    rsum += res['r']
                    psum += res['p']
                p = psum / 3.0
                r = rsum / 3.0
                f1 = 2 * p * r / (p + r)
                f1s.append(f1)
                x_axis.append(rank)
                errors.append(error_dict[m][rank])
        print('"%s": [' % m)
        print('[ ' + ', '.join(map(str, x_axis))+ '],')
        print('[ ' + ', '.join(map(str, f1s)) + ']')
        print('],')
    print('}')
    print('res_err = {')
    for m in methods:
        x_axis = []
        f1s = []
        errors = []
        for rank in range_of_approx:
            predfile = copy.copy(base_pred) % (m, rank)
            if os.path.exists(predfile):
                errors.append(error_dict[m][rank])
                x_axis.append(rank)
        print('"%s": [' % m)
        print('[ ' + ', '.join(map(str, x_axis))+ '],')
        print('[ ' + ', '.join(map(str, errors))+ ']')
        print('],')
    print('}')