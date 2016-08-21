"""
Analysis of detail-free 01 mechanism: what happens if we sample delta from reports, figure
out what's positively correlated, score based on that.

Author: Victor Shnayder <shnayder@gmail.com>
"""
import numpy as np

from collections import Counter
from scipy.stats import rv_discrete

def sample(joint, independent, n=100):
    """
    Take n samples from both joint and independent distributions. Compute
    empirical deltahat.

    Args:
        joint: d-by-d np array, joint prob distribution over two signals
        independent: d-by-d np array, independent prob distribution over two signals

    Returns tuple:
        ("empirical_score": sum(joint-independent where deltahat > 0)/n,
        "ideal_score": sum(joint-independent where delta > 0)/n,
        "sign_wrong": [observed_k != ideal_k for all k] as 1D vector,
        )
    """
    # need to do some heroics to actually flatten matrix to 1-D (not [[a,b,c]])
    joint = np.array(joint.flat)
    ind = np.array(independent.flat)

    K = len(joint)

    delta = joint - ind
    delta_signs = np.sign(delta)

    # make random variables from these distributions
    # http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.rv_discrete.html
    xk = np.arange(K)
    joint_rv = rv_discrete(name='joint', values=(xk, joint))
    ind_rv = rv_discrete(name='ind', values=(xk, ind))

    joint_sample = Counter(joint_rv.rvs(size=n))
    ind_sample = Counter(ind_rv.rvs(size=n))

    empirical_score = sum([joint_sample[k] - ind_sample[k]
                           for k in range(K)
                           if joint_sample[k] > ind_sample[k]])/float(n)
    ideal_score = sum([joint_sample[k] - ind_sample[k]
                           for k in range(K)
                           if joint[k] > ind[k]])/float(n)
    sign_wrong = [np.sign(joint_sample[k] - ind_sample[k]) != np.sign(joint[k]-ind[k])
                  for k in range(K)]

    return (empirical_score, ideal_score, sign_wrong)
