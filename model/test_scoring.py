#!/usr/bin/env python

"""
Tests!

Author: Victor Shnayder <shnayder@gmail.com>
"""

from unittest import TestCase
from world import MultiSignalWorld
from scoring import qsr, logsr, DevilsAdvocateCenter

class TestScoring(TestCase):

    def ensure_calibrated(self, rule):
        # Should return 0 for an uniform prediction
        self.assertEqual(rule([0.5, 0.5], 0), 0.0)
        self.assertEqual(rule([0.5, 0.5], 1), 0.0)
        self.assertEqual(rule([0.25, 0.25, 0.25, 0.25], 0), 0.0)
        self.assertEqual(rule([0.25, 0.25, 0.25, 0.25], 1), 0.0)

        # should return 1 for a perfect prediction
        self.assertEqual(rule([0, 1], 1), 1)
        self.assertEqual(rule([1, 0], 0), 1)
        self.assertEqual(rule([0, 1, 0, 0], 1), 1)

    def test_qsr(self):
        # Make sure it's calibrated
        self.ensure_calibrated(qsr)

    def test_logsr(self):
        self.ensure_calibrated(logsr)


class TestDevilsAdvocateCenter(TestCase):
    def test_payment_bonus(self):
        world = MultiSignalWorld([0.4, 0.6], [[0.8, 0.2],
                                              [0.2, 0.8]],
                                 name="Informed")

        # use a constant base scoring rule to make this easier
        constant_rule = lambda dist, outcome: 2
        epsilon = 0.1
        center = DevilsAdvocateCenter(world, constant_rule,
                                      mode=DevilsAdvocateCenter.ADD_FIXED,
                                      epsilon=epsilon)

        # should have epsilon added
        self.assertEqual(center.payment(1, 0), 2+epsilon)
        self.assertEqual(center.payment(0, 3), 2+epsilon)

        # shouldn't get a bonus
        self.assertEqual(center.payment(0, 0), 2)
        self.assertEqual(center.payment(0, 1), 2)
        self.assertEqual(center.payment(0, 2), 2)
        self.assertEqual(center.payment(1, 1), 2)
        self.assertEqual(center.payment(1, 2), 2)
        self.assertEqual(center.payment(1, 3), 2)


    def test_payment_subtract(self):
        """
        Test the "Follow the Sheep" penalty path.
        """
        world = MultiSignalWorld([0.4, 0.6], [[0.8, 0.2],
                                              [0.2, 0.8]],
                                 name="Informed")

        # use a constant base scoring rule to make this easier
        constant_rule = lambda dist, outcome: 2
        epsilon = 0.1
        center = DevilsAdvocateCenter(world, constant_rule,
                                      mode=DevilsAdvocateCenter.SUB_FIXED,
                                      epsilon=epsilon)

        # should have epsilon subtracted
        self.assertEqual(center.payment(0, 0), 2 - epsilon)
        self.assertEqual(center.payment(1, 3), 2 - epsilon)

        # shouldn't get a bonus
        self.assertEqual(center.payment(0, 1), 2)
        self.assertEqual(center.payment(0, 2), 2)
        self.assertEqual(center.payment(0, 3), 2)
        self.assertEqual(center.payment(1, 0), 2)
        self.assertEqual(center.payment(1, 1), 2)
        self.assertEqual(center.payment(1, 2), 2)

