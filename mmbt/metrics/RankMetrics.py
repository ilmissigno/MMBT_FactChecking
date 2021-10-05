import numpy as np
import torch
from matchzoo.metrics import normalized_discounted_cumulative_gain

PADDED_Y_VALUE = -1 #costant parameter for ndcg

"""
[RankMetrics: This class provides all metrics for Ranking Problems]
[Functions : 
    evaluate_scoreMMBT : Evaluate scores from rankings dictionary
    precision_at_k : Average Precision for Top-K articles
    ndcg_at_k : Normalized Discounted Cumulative Gain for Top-K articles
    dcg_at_k : Discounted Cumulative Gain for Top-K articles
    compute_precision_recall_at_k : Average Precision and Recall for Top-K articles
    HitRatio: Hit Ratio for all articles
    HitRatioAtK: Hit Ratio for Top-K articles
    average_precision : Average Precision for all articles
]
"""

def evaluate_scoreMMBT(rankings, topK):
    """Evaluate scores from rankings dictionary

    Args:
        rankings (Dict): Dictionary per query of docs, labels and similarity scores
        topK (int): K value

    Returns:
        Dict: Dictionary of scores NDCG and HIT
    """
    ndcg_metric = normalized_discounted_cumulative_gain.NormalizedDiscountedCumulativeGain
    hits, ndcgs = [], []
    for query, candidates in rankings.items():
        docs, labels, predictions = candidates
        predictions = np.array(predictions)
        ndcg_mz = ndcg_metric(topK)(labels, predictions)
        ndcgs.append(ndcg_mz)
        positive_docs = set([d for d, lab in zip(docs, labels) if lab == 1])
        indices = np.argsort(-predictions)[:topK]  # indices of items with highest scores
        docs = np.array(docs)
        ranked_docs = docs[indices]
        hit = getHitRatioForList(ranked_docs, positive_docs)
        hits.append(hit)
    results = {}
    results["ndcg"] = np.nanmean(ndcgs)
    results["ndcg_list"] = ndcgs
    results["hits"] = np.nanmean(hits)
    results["hits_list"] = hits
    return results

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE,
         filler_value=1.0):
    """
    Normalized Discounted Cumulative Gain at k.
    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    # y_pred[mask] = 100000.0
    y_pred[mask] = 10000.0
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=-1, index=indices)

def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.
    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[0]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[0], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=0)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:ats_tensor]

    return dcg

def compute_precision_recall(rankedList, k):
    num_hit = np.sum(rankedList[:k])
    precision = float(num_hit) / float(k)
    recall = float(num_hit) / (np.sum(rankedList) + 1e-10)
    return precision, recall

def getHitRatioForList(ranklist, gtItems):
    for item in ranklist:
        if item in gtItems:
            return 1.0
    return 0.0

def getHitRatioAtK(y_pred, y_true, k):
    k = k if k <= y_pred.shape[0] else y_pred.shape[0]
    hits = 0.0

    max_true_id = np.max(y_true[:, 0])
    min_true_id = np.min(y_true[:, 0])

    for i in range(k):
        if y_pred[i, 0] > max_true_id:
            continue
        if y_pred[i, 0] < min_true_id:
            continue
        for j in range(y_true.shape[0]):
            if y_pred[i, 0] == y_true[j, 0]:
                hits += 1.0
                break

    return hits

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
