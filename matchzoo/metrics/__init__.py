from .discounted_cumulative_gain import DiscountedCumulativeGain
from .normalized_discounted_cumulative_gain import \
    NormalizedDiscountedCumulativeGain


def list_available() -> list:
    from matchzoo.engine.base_metric import BaseMetric
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseMetric)
