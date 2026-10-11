"""B plan section 6 map ratios over one reference ROI.

truth: 'occ', 'free' or 'excluded' (boundary band / not measured); predicted: 'occ', 'free' or 'unknown'.
A ratio with an empty denominator is None (undefined), never zero.
"""
import numpy as np

TRUTH = ('occ', 'free', 'excluded')
PREDICTED = ('occ', 'free', 'unknown')


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def map_ratios(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError('Truth and prediction must cover the same nonempty ROI')
    if not np.isin(truth, TRUTH).all() or not np.isin(predicted, PREDICTED).all():
        raise ValueError('Unknown voxel label')
    O, F = truth == 'occ', truth == 'free'
    p_occ, p_free = predicted == 'occ', predicted == 'free'
    known = p_occ | p_free
    return {'false_free_rate': _ratio(np.sum(O & p_free), O.sum()),
            'free_hazard_fraction': _ratio(np.sum(O & p_free), np.sum(p_free & (O | F))),
            'effective_coverage': _ratio(np.sum(known & (O | F)), np.sum(O | F)),
            'false_occupied_rate': _ratio(np.sum(F & p_occ), F.sum()),
            'occupied_recall': _ratio(np.sum(O & p_occ), O.sum()),
            'counts': {'true_occ': int(O.sum()), 'true_free': int(F.sum()),
                       'excluded': int((truth == 'excluded').sum())}}
