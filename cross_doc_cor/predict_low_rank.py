from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from itertools import product
import collections
from tqdm import tqdm

from conll import write_output_file
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from utils import *
from model_utils import *





def init_models(config, device):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(config['model_num'])),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                        "span_scorer_{}".format(config['model_num'])),
                                           map_location=device))
    span_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           "pairwise_scorer_{}".format(config['model_num'])),
                                              map_location=device))
    pairwise_scorer.eval()

    return span_repr, span_scorer, pairwise_scorer



def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    # nested_mentions = collections.defaultdict(list)
    # for i, x in range(len(cluster_ids)):
    #     nested_mentions[x].append(i)

    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]


        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])


    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'


    # Load models and init clustering
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr, span_scorer, pairwise_scorer = init_models(config, device)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])


    # Load data
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    data = create_corpus(config, bert_tokenizer, config.split, is_training=False)

    doc_ids, sentence_ids, starts, ends = [], [], [], []

    # mapping from
    # 'exact' -> []
    # 'symm' -> []
    # 'CUR' -> % rank -> []
    # 'Nystrom' -> % rank -> []
    # 'Nystrom Eig Est' -> % rank -> []

    all_topic_predicted_clusters_dict = {}
    all_scores_dict = {}
    errors_dict = {}
    errors_norm_dict = {}
    max_cluster_id_dict = {}

    # all_topic_predicted_clusters = []
    # max_cluster_id = 0

    # Go through each topic

    for topic_num, topic in enumerate(data.topic_list):
        print('Processing topic {} ({} of {})'.format(topic, topic_num, len(data.topic_list)))
        docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, data, topic_num, docs_embeddings, docs_length)

        doc_id, sentence_id, start, end = span_meta_data
        start_end_embeddings, continuous_embeddings, width = span_embeddings
        width = width.to(device)

        labels = data.get_candidate_labels(doc_id, start, end)

        if config['use_gold_mentions']:
            span_indices = labels.nonzero().squeeze(1)
        else:
            k = int(config['top_k'] * num_of_tokens)
            with torch.no_grad():
                span_emb = span_repr(start_end_embeddings, continuous_embeddings, width)
                span_scores = span_scorer(span_emb)
            _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)
            # span_indices, _ = torch.sort(span_indices)


        number_of_mentions = len(span_indices)
        start_end_embeddings = start_end_embeddings[span_indices]
        continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
        width = width[span_indices]
        torch.cuda.empty_cache()

        # Prepare all the pairs for the distance matrix
        first, second = zip(*list(product(range(len(span_indices)), repeat=2)))
        first = torch.tensor(first)
        second = torch.tensor(second)

        torch.cuda.empty_cache()
        all_scores = []
        with torch.no_grad():
            # g1 = span_repr(start_end_embeddings[first],
            #                [continuous_embeddings[k] for k in first],
            #                width[first])
            # g2 = span_repr(start_end_embeddings[second],
            #                [continuous_embeddings[k] for k in second],
            #                width[second])
            #
            # scores = pairwise_scorer(g1, g2)
            # scores = torch.sigmoid(scores)
            # all_scores.csv.extend(scores.squeeze(1))
            # torch.cuda.empty_cache()

            for i in range(0, len(first), 10000):
                # end_max = min(i+100000, len(first))
                end_max = i+10000
                first_idx, second_idx = first[i:end_max], second[i:end_max]
                g1 = span_repr(start_end_embeddings[first_idx],
                               [continuous_embeddings[k] for k in first_idx],
                               width[first_idx])
                g2 = span_repr(start_end_embeddings[second_idx],
                               [continuous_embeddings[k] for k in second_idx],
                               width[second_idx])
                scores = pairwise_scorer(g1, g2)

                torch.cuda.empty_cache()
                if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                    g1_score = span_scorer(g1)
                    g2_score = span_scorer(g2)
                    scores += g1_score + g2_score

                scores = torch.sigmoid(scores)
                all_scores.extend(scores.detach().cpu().squeeze(1))
                torch.cuda.empty_cache()

        all_scores = torch.stack(all_scores)
        sims = all_scores.view(number_of_mentions, number_of_mentions).numpy()
        print('sims %s' % str(sims.shape))
        methods = ['exact', 'sym', 'nystrom', 'cur', 'cur_alt', 'nystrom_eig_est_rescale', 'nystrom_eig_est', 'sms_nystrom', 'sms_nystrom_rescale', 'skeleton']
        range_of_approx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for m in methods:
            if m == 'exact':
                approx_sims = sims
                rank = 1.0
                pairwise_distances = 1 - approx_sims
                predicted = clustering.fit(pairwise_distances)
                if m not in max_cluster_id_dict:
                    max_cluster_id_dict[m] = {rank: 0}
                predicted_clusters = predicted.labels_ + max_cluster_id_dict[m][rank]
                if m not in all_topic_predicted_clusters_dict:
                    all_topic_predicted_clusters_dict[m] = {rank: []}
                all_topic_predicted_clusters_dict[m][rank].extend(predicted_clusters)

                max_cluster_id_dict[m][rank] = max(predicted_clusters) + 1
                error = 0.0
                if m not in errors_dict:
                    errors_dict[m] = {rank: 0.0}
                    errors_norm_dict[m] = {rank: 0}
                errors_dict[m][rank] += error
                errors_norm_dict[m][rank] += sims.shape[0]*sims.shape[1]
            elif m == 'sym':
                approx_sims = 0.5 * (sims + sims.T)
                print('approx_sims=%s' % (str(approx_sims.shape)))
                rank = 1.0
                pairwise_distances = 1 - approx_sims
                predicted = clustering.fit(pairwise_distances)
                if m not in max_cluster_id_dict:
                    max_cluster_id_dict[m] = {rank: 0}
                predicted_clusters = predicted.labels_ + max_cluster_id_dict[m][rank]
                if m not in all_topic_predicted_clusters_dict:
                    all_topic_predicted_clusters_dict[m] = {rank: []}
                all_topic_predicted_clusters_dict[m][rank].extend(predicted_clusters)

                max_cluster_id_dict[m][rank] = max(predicted_clusters) + 1

                error = np.linalg.norm(approx_sims-sims) **2
                if m not in errors_dict:
                    errors_dict[m] = {rank: 0.0}
                    errors_norm_dict[m] = {rank: 0}
                errors_dict[m][rank] += error
                errors_norm_dict[m][rank] += np.linalg.norm(sims) **2

            else:
                from low_rank import nystrom_with_eig_estimate_sub, CUR, nystrom, CUR_alt, nystrom_with_eig_estimate
                for rank in range_of_approx:
                    print('m=%s rank=%s' % (m, rank))
                    symm = 0.5 * (sims + sims.T)
                    print('approx_sims=%s' % (str(symm.shape)))

                    if m == 'cur':
                        approx_sims = CUR(symm, k=int(rank*sims.shape[0]))
                    elif m == 'cur_alt':
                        approx_sims = CUR_alt(symm, k=int(rank*sims.shape[0]))
                    elif m == 'nystrom_eig_est_rescale':
                        approx_sims = nystrom_with_eig_estimate(symm, int(rank*sims.shape[0]), new_rescale=True)
                    elif m == 'nystrom_eig_est':
                        approx_sims = nystrom_with_eig_estimate(symm, int(rank*sims.shape[0]), new_rescale=False)
                    elif m == 'sms_nystrom_rescale':
                        approx_sims = nystrom_with_eig_estimate_sub(symm, int(rank*sims.shape[0]), new_rescale=True)
                    elif m == 'sms_nystrom':
                        approx_sims = nystrom_with_eig_estimate_sub(symm, int(rank * sims.shape[0]), new_rescale=False)
                    elif m == 'nystrom':
                        approx_sims = nystrom(symm, int(rank * sims.shape[0]))
                    elif m == 'skeleton':
                        approx_sims = CUR_alt(symm, int(rank * sims.shape[0]), mult=1)

                    print('max: %s' % np.max(approx_sims))
                    print('min: %s' % np.min(approx_sims))
                    print('mean: %s' % np.mean(approx_sims))
                    print('std: %s' % np.std(approx_sims))
                    pairwise_distances = 1 - approx_sims
                    predicted = clustering.fit(pairwise_distances)
                    if m not in max_cluster_id_dict:
                        max_cluster_id_dict[m] = {}

                    if rank not in max_cluster_id_dict[m]:
                        max_cluster_id_dict[m][rank] = 0
                    predicted_clusters = predicted.labels_ + max_cluster_id_dict[m][rank]
                    if m not in all_topic_predicted_clusters_dict:
                        all_topic_predicted_clusters_dict[m] = {}
                    if rank not in all_topic_predicted_clusters_dict[m]:
                        all_topic_predicted_clusters_dict[m][rank] = []
                    all_topic_predicted_clusters_dict[m][rank].extend(predicted_clusters)

                    max_cluster_id_dict[m][rank] = max(predicted_clusters) + 1

                    error = np.linalg.norm(symm - approx_sims) ** 2
                    if m not in errors_dict:
                        errors_dict[m] = {}
                        errors_norm_dict[m] = {}
                    if rank not in errors_dict[m]:
                        errors_dict[m][rank] = 0.0
                        errors_norm_dict[m][rank] = 0.0
                    errors_dict[m][rank] += error
                    errors_norm_dict[m][rank] += np.linalg.norm(symm) ** 2
                    if m not in all_scores_dict:
                        all_scores_dict[m] = dict()
                    if rank not in all_scores_dict[m]:
                        all_scores_dict[m][rank] = dict()
                    all_scores_dict[m][rank][topic_num] = approx_sims


            # # approx_sims = sims
            # print('size of sims %s' % str(sims.shape))
            # rank = int(0.9*sims.shape[0])
            # approx_sims = nystrom_with_eig_estimate(sims, k=rank)
            # # approx_sims = CUR(sims, k=rank)
            # # approx_sims = np.real(approx_sims)
            # # Affinity score to distance score
            # pairwise_distances = 1 - approx_sims
            # predicted = clustering.fit(pairwise_distances)
            # predicted_clusters = predicted.labels_ + max_cluster_id
            # max_cluster_id = max(predicted_clusters) + 1
            # all_topic_predicted_clusters.extend(predicted_clusters)

        doc_ids.extend(doc_id[span_indices.cpu()])
        sentence_ids.extend(sentence_id[span_indices].tolist())
        starts.extend(start[span_indices].tolist())
        ends.extend(end[span_indices].tolist())
        torch.cuda.empty_cache()


    os.makedirs(config['save_path'] + '/low_rank', exist_ok=True)

    final_error_dict = dict()
    for m in errors_dict:
        final_error_dict[m] = dict()
        for rank in errors_dict[m]:
            err = np.sqrt(errors_dict[m][rank]) / np.sqrt(errors_norm_dict[m][rank])
            final_error_dict[m][rank] = err

    print(final_error_dict)

    with open(config['save_path'] + '/low_rank/results_low_rank.pkl', 'wb') as fout:
        pickle.dump([final_error_dict, all_scores_dict, errors_dict, errors_norm_dict, all_topic_predicted_clusters_dict, max_cluster_id_dict], fout)
    doc_ids_orig, starts_orig, ends_orig = doc_ids, starts, ends
    for m in all_topic_predicted_clusters_dict:
        for rank in all_topic_predicted_clusters_dict[m]:
            print('m=%s rank %s' % (m, rank))
            all_topic_predicted_clusters = all_topic_predicted_clusters_dict[m][rank]
            all_clusters = {}
            for i, cluster_id in enumerate(all_topic_predicted_clusters):
                if cluster_id not in all_clusters:
                    all_clusters[cluster_id] = []
                all_clusters[cluster_id].append(i)

            if not config['use_gold_mentions']:
                all_clusters, doc_ids, starts, ends = remove_nested_mentions(all_clusters, doc_ids_orig.copy(), starts_orig.copy(), ends_orig.copy())

            all_clusters = {cluster_id:mentions for cluster_id, mentions in all_clusters.items() if len(mentions) > 1}

            print('Saving conll file...')
            doc_name = '{}_{}_{}_{}_model_{}'.format(
                config['split'], config['mention_type'], config['linkage_type'], config['threshold'], config['model_num'])

            write_output_file(data.documents, all_clusters, doc_ids, starts, ends, config['save_path'] + '/low_rank/%s/%s/' % (m,rank), doc_name,
                              topic_level=config.topic_level, corpus_level=not config.topic_level)