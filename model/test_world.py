#!/usr/bin/env python
"""
Tests!

Author: Victor Shnayder <shnayder@gmail.com>
"""

from math import factorial

from unittest import TestCase
from world import MultiSignalWorld

# the world from the Jurca-Faltings '09 paper
# asymmetric, so should be good.
# p0 = 0.2
# p1 = 0.8
# p00 = 1.0   # p of 0 given true state 0
# p10 = 0.0   # p of 1 given true state 0
# p01 = 0.1
# p11 = 0.9
p0 = 0.2
p1 = 0.8
p00 = 0.85   # p of 0 given true state 0
p10 = 0.15   # p of 1 given true state 0
p01 = 0.1
p11 = 0.9

def n_choose_k(n, k):
    return factorial(n) / (factorial(k) * factorial(n-k))


class TestWorld(TestCase):

    longMessage = True
    def setUp(self):
        self.jf_world = MultiSignalWorld([p0, p1],
                            [[p00, p01],
                             [p10, p11]],
                    name="JF")

    def aae(self, *args, **kwargs):
        self.assertAlmostEqual(*args, **kwargs)

    def test_prob_signal(self):
        self.aae(self.jf_world.prob_signal(0),
                         p0 * p00 + p1*p01)
        self.aae(self.jf_world.prob_signal(1),
                         p0 * p10 + p1*p11)

    def test_prob_signal_pair(self):
        self.aae(self.jf_world.prob_signal_pair(0, 0),
                         p0 * (p00 * p00) + p1 * (p01 * p01))
        self.aae(self.jf_world.prob_signal_pair(0, 1),
                         p0 * (p00 * p10) + p1 * (p01 * p11))
        self.aae(self.jf_world.prob_signal_pair(1, 0),
                 self.jf_world.prob_signal_pair(0, 1))
        # and insist that they sum to one
        self.aae(sum(self.jf_world.prob_signal_pair(a,b)
                     for a in [0,1]
                     for b in [0,1]), 1.0)


    def test_posterior_others_signal(self):
        """
        Compute the posterior probability that another agent
        got others_signal, given that you got the observed signal.
        """
        # (Assume above functions work, since they're tested)

        # posterior_others_signal(observed, others_signal)
        for obs in [0, 1]:
            for signal in [0, 1]:
                self.aae(self.jf_world.posterior_others_signal(obs, signal),
                         self.jf_world.prob_signal_pair(obs, signal) / self.jf_world.prob_signal(obs))

        self.aae(self.jf_world.posterior_others_signal(0, 1),
                 (p0 * p00 * p10 + p1 * p01 * p11) / (p0 * p00 + p1 * p01))



    def test_prob_state_given_signal(self):
        """
        prob_state_given_signal(self, state, signal):
        Return the probability that the true state is state given an observed
        signal.  (Essentially, apply Bayes rule to model)
        """
        for state in [0,1]:
            for signal in [0,1]:
                self.aae(self.jf_world.prob_state_given_signal(state, signal),
                         (self.jf_world.conditional[signal][state] *
                          self.jf_world.prior[state] / self.jf_world.prob_signal(signal)))

        self.aae(self.jf_world.prob_state_given_signal(0, 0),
                 p00 * p0 / (p00 * p0 + p01 * p1))


    def test_prob_signal_count(self):
        """
        prob_signal_count(self, count, N, other_signal, my_signal):

        Return the probability that count out of N-1 reference people will see signal,
        given that I saw obs.
        """
        p_of_count = self.jf_world.prob_signal_count

        # These numbers match the JF paper, though they were
        # computed by my code, so the only purpose of the test
        # is to make sure that it doesn't break later.
        actual = [p_of_count(i, 4, 1, 0) for i in range(4)]
        expected = [0.417925, 0.229725, 0.116775, 0.235575]

        for act, exp in zip(actual, expected):
            self.aae(act, exp)


