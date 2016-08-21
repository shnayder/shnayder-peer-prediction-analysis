"""
Tests!

Author: Victor Shnayder <shnayder@gmail.com>
"""

from unittest import TestCase

from world import MultiSignalWorld, get_binary_world
from scoring import (ScoringRuleCenter, DevilsAdvocateCenter, qsr)

from analysis import (is_truthful_equilibrium, is_fixed_report_equilibrium,
                      probs_1s_given_signal)

informed_world = MultiSignalWorld([0.4, 0.6], [[0.8, 0.2],
                                               [0.2, 0.8]],
                                               name="Informed")

very_informed_world = MultiSignalWorld([0.4, 0.6], [[0.99, 0.01],
                                            [0.01, 0.99]],
                               name="Very Informed")
uninformed_world = MultiSignalWorld([0.4, 0.6], [[0.51, 0.49],
                                         [0.49, 0.51]],
                            name="Uninformed")

jf_world = MultiSignalWorld([0.2, 0.8],
                            [[0.85, 0.1],
                             [0.15, 0.9]],
                    name="JF")



class TestEquilibriumComputation(TestCase):
    def test_is_truthful_equilibrium(self):

        def check_world(world):
            return is_truthful_equilibrium(ScoringRuleCenter(world, qsr))


        # for qsr, any world should give a truthful equilibrium
        self.assertTrue(check_world(informed_world))
        self.assertTrue(check_world(very_informed_world))
        self.assertTrue(check_world(uninformed_world))
        self.assertTrue(check_world(jf_world))

        # just check a bunch of random worlds
        self.assertTrue(check_world(get_binary_world(0.2, 0.3, 0.4)))
        self.assertTrue(check_world(get_binary_world(0.8, 0.01, 0.99)))

        # what about weird ones where p(H|L) > p(H|H)
        self.assertTrue(check_world(get_binary_world(0.8, 0.8, 0.4)))

        def antiqsr(*args):
            return -qsr(*args)

        def anticheck(world):
            return is_truthful_equilibrium(ScoringRuleCenter(world, antiqsr))

        # if the scoring rule is negated, should have the opposite effect--nothing
        # should be an equilibrium

        self.assertFalse(anticheck(informed_world))
        self.assertFalse(anticheck(very_informed_world))
        self.assertFalse(anticheck(uninformed_world))
        self.assertFalse(anticheck(jf_world))


class TestHelperFunctions(TestCase):
    def test_probs_1s_given_signal(self):
        # Expected numbers confirmed from JF paper
        actual = probs_1s_given_signal(jf_world, 0, 4)
        expected = [0.417925, 0.229725, 0.116775, 0.235575]

        for act, exp in zip(actual, expected):
            self.assertAlmostEqual(act, exp)
