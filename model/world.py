"""
'World' encapsulates a signal model and some handy functions...

Author: Victor Shnayder <shnayder@gmail.com>
"""
from copy import deepcopy
from random import random
import math
import numpy as np

from parameters import Parameters


class BinaryWorld(object):
    """Encapsulate parameters for a binary signal world--shouldn't be used anymore"""
    def __init__(self, prior_high, prob_high_given_high, prob_high_given_low):
        self.prior_high = prior_high
        self.prob_high_given_high = prob_high_given_high
        self.prob_high_given_low = prob_high_given_low
        self.prior = [1.0 - prior_high, prior_high]
        # conditional of i given t: conditional[i][t]
        self.conditional = [[1.0 - prob_high_given_low,
                             1.0 - prob_high_given_high],
                            [prob_high_given_low,
                             prob_high_given_high]]

    @property
    def prior_low(self):
        return 1.0 - self.prior_high

    @property
    def prob_low_given_high(self):
        return 1.0 - self.prob_high_given_high

    @property
    def prob_low_given_low(self):
        return 1.0 - self.prob_high_given_low

    def posterior_others_signal_high(self, observed):
        """
        Compute the posterior probability that another agent
        got a high signal, given the observed signal.
        """
        ph = self.prior_high
        pl = self.prior_low
        phh = self.prob_high_given_high
        phl = self.prob_high_given_low
        plh = self.prob_low_given_high
        pll = self.prob_low_given_low

        # define prob_observed_given_high, low
        poh = self.conditional[observed][1]
        pol = self.conditional[observed][0]

        return (phh*poh*ph + phl*pol*pl)/(poh*ph + pol*pl)

    def prob_signal_pair(self, s1, s2):
        """
        Return the probability of two agents getting the signal
        pair s1, s2
        """
        ph = self.prior_high
        pl = self.prior_low
        def p(obs, real):
            """prob of observed given true state"""
            return self.conditional[obs][real]

        return ph * p(s1, 1) * p(s2, 1) + pl * p(s1, 0) * p(s2, 0)

    def prob_signal(self, s):
        """
        Return the probability of an agent getting the signal s.
        """
        ph = self.prior_high
        pl = self.prior_low
        def p(obs, real):
            """prob of observed given true state"""
            return self.conditional[obs][real]

        return ph * p(s, 1) + pl * p(s, 0)

    def __str__(self):
        return """P(h) = {ph:4.2}, P(l) = {pl:4.2}
P(h|h) = {phh:4.2}   P(l|h) = {plh:4.2}
P(h|l) = {phl:4.2}   P(l|l) = {pll:4.2}""".format(
            ph=self.prior_high,
            pl=self.prior_low,
            phh=self.prob_high_given_high,
            phl=self.prob_high_given_low,
            plh=self.prob_low_given_high,
            pll=self.prob_low_given_low)


def nCk(n, k):
    """
    Compute n choose k.
    """
    f = math.factorial
    return f(n) / f(k) / f(n-k)


def get_binary_world(prior_high, prob_high_given_low, prob_high_given_high):
        prior = [1.0 - prior_high, prior_high]
        # conditional of signal s given t: conditional[s][t]
        conditional = [[1.0 - prob_high_given_low, 1.0 - prob_high_given_high],
                       [prob_high_given_low, prob_high_given_high]]
        return MultiSignalWorld(prior, conditional)


def get_random_worlds(k):
    """
    Return a list of k binary worlds with random parameters (uniformly chosen).
    """
    return [get_binary_world(random(), random(), random())
            for i in range(k)]


def random_dist(k):
    """
    Return a random distribution over k elements.
    """
    # hopefully non-biased algorithm:
    # pick k random numbers
    # normalize
    dist = np.random.rand(k)
    dist /= np.sum(dist)
    return dist

def get_random_multisignal_world(n):
    """
    Return a random world with n signals.
    """
    #
    prior = list(random_dist(n)[:-1])
    cond = [list(random_dist(n)[:-1])
            for i in range(n)]
    return multisignal_world_alt(prior, cond)


def multisignal_world_alt(prior, cond, name=None):
    """
    Alternate constructor for MultiSignalWorld.

    prior: array of n-1 priors. The remaining one will be 1-sum(prior).
    cond: array of n conditionals, given true type. cond[t][s] is P(s|t).
        each conditional should have length n-1. P(n|t) will be 1-sum(P(i|t))
    name: optional. Name for the world.
    """
    def check_dist(x, val_str):
        if x > 1:
            raise ValueError("{} = {} > 1".format(val_str, x))

    check_dist(sum(prior), "sum(prior)")
    full_prior = prior + [1 - sum(prior)]

    n = len(full_prior)
    full_cond = [[None for i in range(n)]
                 for j in range(n)]
    for t in range(n):
        for s in range(n-1):
            full_cond[s][t] = cond[t][s]

        check_dist(sum(cond[t]), "sum(cond[t])")
        full_cond[n-1][t] = 1 - sum(cond[t])

    return MultiSignalWorld(full_prior, full_cond, name)



class MultiSignalWorld(object):
    """Encapsulate parameters for a world with many signals"""
    def __init__(self, prior, conditional, name=None):
        """
        prior -- a list of the priors for signals: prior[i] is the prior for true
              state i.  Must add up to 1.0
        conditional -- conditional[s][t] is P(signal s|true state t).
              conditional[][t] must add up to 1.0
        """
        if abs(sum(prior) - 1.0) > 0.00001:
            raise ValueError("prior {} must add up to 1.0".format(prior))
        n = len(prior)
        for t in range(n):
            if abs(sum(conditional[i][t] for i in range(n)) - 1.0) > 0.00001:
                raise ValueError("conditional[][t] must add up to 1.0"
                                 " for t={} (got {})".format(t, conditional))

        # num states
        self.n = len(prior)
        self.prior = prior
        self.conditional = conditional
        self.name = name or ""


    def prob_signal(self, s):
        """
        Return the probability of an agent getting the signal s.
        """
        return sum(self.prior[i] * self.conditional[s][i]
                   for i in range(self.n))


    def prob_signal_pair(self, s1, s2):
        """
        Return the probability of two agents getting the signal
        pair s1, s2
        """
        def p(obs, real):
            """prob of observed given true state"""
            return self.conditional[obs][real]

        # add up prob_truth * conditional_likelyhood_of_s1 * ditto_of_s2
        return sum(self.prior[i] * p(s1, i) * p(s2, i)
                   for i in range(self.n))


    def prob_count(self, count, N, signal):
        """
        Return the probability that count out of N people will see signal.
        """
        def prob_count_given_state(state):
            return (nCk(N, count) *
                    self.conditional[signal][state]**count *
                    (1 - self.conditional[signal][state])**(N - count))

        return sum(prob_count_given_state(state) * self.prior[state]
                   for state in range(self.n))


    def posterior_others_signal(self, observed, others_signal):
        """
        Compute the posterior probability that another agent
        got others_signal, given that you got the observed signal.
        """
        return self.prob_signal_pair(observed, others_signal) / self.prob_signal(observed)


    def prob_state_given_signal(self, state, signal):
        """
        Return the probability that the true state is state given an observed
        signal.  (Essentially, apply Bayes rule to model)
        """
        return (self.conditional[signal][state] * self.prior[state] /
                    self.prob_signal(signal))

    def prob_signal_count(self, count, N, other_signal, my_signal):
        """
        Return the probability that count out of N-1 reference people will see
        other_signal, given that I saw my_signal.
        """
        def prob_count_given_state(state):
            return (nCk(N - 1, count) *
                    self.conditional[other_signal][state]**count *
                    (1 - self.conditional[other_signal][state])**(N - 1 - count))

        return sum(prob_count_given_state(state) *
                   self.prob_state_given_signal(state, my_signal)
                   for state in range(self.n))

    def joint_matrix(self):
        """
        Return the matrix of joint probs: entry i,j is P(s1=i,s2=j|t1=t2)

        Returns a numpy array.
        """
        a = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                a[i,j] = self.prob_signal_pair(i,j)
        return a


    def ind_pair_matrix(self):
        """
        Return the matrix of independent probs: entry i,j is P(s1=i)P(s2=j)

        Returns a numpy array.
        """
        a = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                a[i,j] = self.prob_signal(i)*self.prob_signal(j)
        return a


    def delta_matrix(self):
        """
        Return the Delta matrix: entry i,j is P(s1=i,s2=j|t1=t2) - P(s1=i)P(s2=j).

        Returns a numpy array.
        """
        return self.joint_matrix() - self.ind_pair_matrix()


    def __str__(self):
        def format_lst(lst):
            return ", ".join("{:4.2}".format(x) for x in lst)

        p = format_lst(self.prior)
        cs = [format_lst(lst) for lst in self.conditional]
        return """Priors:
{}
Conditionals:
{}
""".format(p, "\n".join(cs))


def is_self_predicting(world_or_joint, verbose=False):
    """
    If called with a MultiSignalWorld object, uses it's joint. If called with an nxn np
    matrix, assumes it's the joint.

    if verbose, prints where self-predicting condition fails
    """
    if isinstance(world_or_joint, MultiSignalWorld):
        joint = world_or_joint.joint_matrix()
    else:
        joint = world_or_joint

    prior = np.array(joint.sum(0).flat) # assume symmetry
    if verbose:
        print prior

    def post(s_given_peer, s_me):
        """Posterior prob of another agent's signal given mine"""
        #        print joint, prior, s_me, s_given_peer
        return joint[s_me, s_given_peer] / prior[s_me]


    n = joint.shape[0]
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            if post(x,y) > post(x,x):
                if verbose:
                    print "Fail at x,y={},{}, post(x|y)={} > post(x|x)={}".format(
                        x, y, post(x,y), post(x,x))
                return False

    return True
