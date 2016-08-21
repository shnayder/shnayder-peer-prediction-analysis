"""
Mostly old functions from when I was trying variations on peer shadowing and JF09.

The basic scoring functions are still being used.

Author: Victor Shnayder <shnayder@gmail.com>
"""

from collections import namedtuple
from math import log

import matplotlib.pyplot as plt

from analysis import probs_1s_given_signal

def qsr(predicted_dist, outcome):
    """
    Compute the quadratic scoring rule, normalized so that perfect predictions
    get a score of 1, and uniform predictions a score of 0.

    Args:
        predicted_dist: the predicted distribution (sums to 1.0) of probabilities for
                        possible outcomes.
        outcome: the actual outcome--an index into that array.
    """
    p_i = predicted_dist[outcome]
    norm = sum([x*x for x in predicted_dist])
    n = float(len(predicted_dist))

    # multiply by n/(n-1) to make the range 1 when going from uniform to 1,
    # subtract n/(n-1)*1/n to make it zero at uniform.
    return n/(n-1) * (2*p_i - norm - 1/n)


def qsr_norm_0_to_1(predicted_dist, outcome):
    """
    Compute the quadratic scoring rule, normalized so that the scores are between 0 and 1.

    Args:
        predicted_dist: the predicted distribution (sums to 1.0) of probabilities for
                        possible outcomes.
        outcome: the actual outcome--an index into that array.
    """
    p_i = predicted_dist[outcome]
    norm = sum([x*x for x in predicted_dist])

    # base qsr is between -1 and 1, so adjust
    return 0.5 + (2*p_i - norm)/2.0


def logsr(predicted_dist, outcome):
    """
    Compute the logarithmic scoring rule, normalized so that perfect predictions
    get a score of 1, and uniform predictions a score of 0.

    Args:
        predicted_dist: the predicted distribution (sums to 1.0) of probabilities for
                        possible outcomes.
        outcome: the actual outcome--an index into that array.

    If predicted_dist[outcome] is 0, this scoring rule is not defined, and will
    raise ValueError
    """
    n = float(len(predicted_dist))
    # divide by -log(1/n) to normalize range, add 1 to make it zero at uniform
    return log(predicted_dist[outcome]) / -log(1/n) + 1




class ScoringRuleCenter(object):
    """
    A mechanism center for a world, using a version of the MRZ mechanism:
    with multiple reference raters, compute the posterior distribution over
    their reports, score against that.

    NOTE: Only works for binary for now.
    """
    def __init__(self, world, scoring_rule, N=4):
        """
        Args:
            world: a MultiSignalWorld (though only using binary for now)
            scoring_rule: a function that takes (predicted_dist, outcome)
                and returns a float.
            N: optional (default 4).  How many agents play in each round.
               If >2, use more than one reference rater.
        """
        self.world = world
        self.scoring_rule = scoring_rule
        self.N = N


    def payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer,
        compute the implied posterior distribution given that report,
        and score it using the scoring rule.
        """
        implied_distribution = probs_1s_given_signal(self.world, report, self.N)
        return self.scoring_rule(implied_distribution, outcome)


Line = namedtuple("Line", ["slope", "intercept"])
Point = namedtuple("Point", ["x", "y"])


def intersect(line1, line2):
    """
    Return the intersection of two lines, in slope+intercept form,
    or None if there isn't one (parallel lines).

    Returns a Point.
    """
    if line1.slope - line2.slope == 0:
        return None

    isect_x = (line2.intercept - line1.intercept) / (line1.slope - line2.slope)
    isect_y = line1.slope * isect_x + line1.intercept
    return Point(isect_x, isect_y)


def line_func(line):
    return lambda x: line.slope * x + line.intercept


def expand_range(t, fraction=0.2):
    """
    Given an interval (low, high), return an interval [lower, higher],
    so that the length of the interval is increased by fraction.
    """
    low, high = t
    length = high - low
    f = fraction / 2.0
    return [low - length*f, high + length*f]


def span(*args):
    """
    span(a,b,c) = (min(a,b,c), max(a,b,c))
    """
    return min(*args), max(*args)


class DevilsAdvocateCenter(object):
    """
    A mechanism center for a world, using a version of the MRZ mechanism
    with a twist:
    with multiple reference raters, compute the posterior distribution over
    their reports, score against that, then give a bonus for going against
    the grain.

    NOTE: Only works for binary for now.
    """

    # constants for various modes of operation
    ADD_FIXED = 0
    SUB_FIXED = 1
    ADAPTIVE = 2

    def __init__(self, world, scoring_rule, mode=ADAPTIVE, epsilon=0.1, N=4):
        """
        Args:
            world: a MultiSignalWorld (though only using binary for now)
            scoring_rule: a function that takes (predicted_dist, outcome)
                and returns a float.
            mode: add a Devil's Advocate bonus of epsilon if ADD_FIXED,
                  subtract a Follow the Sheep penalty of epsilon if SUB_FIXED,
                  do a magical thing if ADAPTIVE :)
            epsilon: How much to add or subtract from the reference payment.
            N: optional (default 4).  How many agents play in each round.
               If >2, use more than one reference rater.
        """
        self.world = world
        self.scoring_rule = scoring_rule
        self.N = N
        self.epsilon = epsilon
        self.mode = mode
        self._precompute()


    def _base_payment(self, report, outcome):
        """
        Return the payment using just the scoring rule.
        """
        implied_distribution = probs_1s_given_signal(self.world, report, self.N)
        return self.scoring_rule(implied_distribution, outcome)


    def _added_payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer,
        compute the implied posterior distribution given that report,
        and score it using the scoring rule. If this is a "Devil's Advocate"
        report, make the payment the base payment for agreement plus epsilon
        (to make all-1s and all-0s not an equilibrium).
        """
        base_score = self._base_payment(report, outcome)
        if report == 1 and outcome == 0:
            return self._base_payment(0, 0) + self.epsilon

        if report == 0 and outcome == self.N - 1:
            return self._base_payment(1, self.N - 1) + self.epsilon

        return base_score


    def _subtracted_payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer, compute the
        implied posterior distribution given that report, and score it using the scoring
        rule. If this is a "We all agree" report, make the payment the base payment for
        being a devil's advocate minus epsilon (to make all-1s and all-0s not an
        equilibrium).
        """
        base_score = self._base_payment(report, outcome)
        if report == 0 and outcome == 0:
            return self._base_payment(1, 0) - self.epsilon

        if report == 1 and outcome == self.N - 1:
            return self._base_payment(0, self.N - 1) - self.epsilon

        return base_score


    def _expected_base_payment_given_signal_report(self, signal, report):
        """
        Given that I saw a particular signal, what's the expected base payment
        for reporting report (given others are truthful).
        """
        return sum(self._base_payment(report, num_ones) *
                   self.world.prob_signal_count(num_ones, self.N, 1, signal)
                   for num_ones in range(self.N))


    def _prob_num_ones(self, signal, count):
        """
        Return prob of count 1s given that I saw signal.
        """
        return self.world.prob_signal_count(count, self.N, 1, signal)


    def compute_constraints(self):
        """
        Return the two lines and min_x and min_y constraints for this world and payment
        rule, in a form that's easy to plot.

        Returns: (line1, line2, min_x, min_y): (Line, Line, float, float)
        """

        # we'll want some convenient terms for relevant base payments:
        # sig \ #1s  0   1  ... N-1
        #    0  |    a           b
        #    1  |    c           d

        a = self._base_payment(0, 0)
        b = self._base_payment(0, self.N-1)
        c = self._base_payment(1, 0)
        d = self._base_payment(1, self.N-1)

        # The "gaps" between expected value between being truthful and misreporting,
        # for each signal.
        delta0 = (self._expected_base_payment_given_signal_report(0, 0) -
                  self._expected_base_payment_given_signal_report(0, 1))

        delta1 = (self._expected_base_payment_given_signal_report(1, 1) -
                  self._expected_base_payment_given_signal_report(1, 0))

        # we'll need some conditional probs too
        p = self._prob_num_ones

        # debugging
        self.delta0, self.delta1 = delta0, delta1
        self.a, self.b, self.c, self.d = (a, b, c, d)

        return (Line(p(0,0)/p(0,3), -self.delta0/p(0,3)), Line(p(1,0)/p(1,3), self.delta1/p(1,3)),
                self.a-self.c, self.d-self.b)


    def _precompute(self):
        # compute the bonuses to add for 0 and N-1:

        constraints = (line0, line1, min_x, min_y) = self.compute_constraints()
        # critical point
        crit_pt = intersect(line0, line1)
        if crit_pt is None:
            # no intersection. Give up.

            # debugging
            self.x, self.y, self.x_slack, self.y_slack, self.x_ok, self.y_ok = (
                None, None, None, None, False, False)

            self.bonus_for_zero = 0
            self.bonus_for_N = 0

            return

        x, y = crit_pt
        x_slack = x - min_x
        y_slack = y - min_y

        # debugging
        self.x, self.y, self.x_slack, self.y_slack, self.x_ok, self.y_ok = (
            x, y, x_slack, y_slack, x_slack > 0, y_slack > 0)

        if x_slack < 0 or y_slack < 0:
            # can't make it work.
            self.bonus_for_zero = 0
            self.bonus_for_N = 0
            return

        # Otherwise, we can make it work: compute a point off the critical path
        min_line = Line(0, min_y)

        isect = intersect(line0, min_line)
        if isect.x > min_x:
            mid_x = (isect.x + x)/2
        else:
            mid_x = (min_x + x)/2

        pt = Point(mid_x, (line_func(line0)(mid_x) + line_func(line1)(mid_x))/2)
        self.bonus_for_zero = pt.x
        self.bonus_for_N = pt.y



    def _magic_payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer, compute the
        implied posterior distribution given that report, and score it using the scoring
        rule. If this is a "Devil's advocate" report, make the payment the base payment
        for agreement plus an adaptive bonus, chosen to make 0 and 1 have the same
        relative expected value.
        """
        base_score = self._base_payment(report, outcome)
        if report == 1 and outcome == 0:
            return self._base_payment(1, 0) + self.bonus_for_zero

        if report == 0 and outcome == self.N - 1:
            return self._base_payment(0, self.N - 1) + self.bonus_for_N

        return base_score


    def payment(self, report, outcome):
        if self.mode == self.ADD_FIXED:
            return self._added_payment(report, outcome)
        elif self.mode == self.SUB_FIXED:
            return self._subtracted_payment(report, outcome)
        else:
            return self._magic_payment(report, outcome)


def alpha_beta_lines(world, N=4):
    """
    Return two lines: line0, which you must be above, and line 1, which you must be below
    """
    p = lambda signal, count: world.prob_signal_count(count, N, 1, signal)
    return (Line(p(0,N-2) / p(0,1), (p(0,0) - p(0,N-1))/p(0,1)),
            Line(p(1,N-2) / p(1,1), (p(1,0)-p(1,N-1))/p(1,1)))


def compute_alpha_beta(world, N=4):
    """
    Returns a tuple (alpha, beta)
    """
    line0, line1 = alpha_beta_lines(world, N)
    isect = intersect(line0, line1)
    if isect is None:
        return (-1, -1)

    # critical point, with beta on x axis
    (beta_0, alpha_0) = isect
    # go out to the right a bit, and pick a point between the lines
    beta = x = beta_0 + 10
    alpha = (line_func(line0)(x) + line_func(line1)(x))/2
    return alpha, beta


def _alpha_beta_payment(alpha, beta, N, report, outcome, epsilon=1):
    if report==0:
        if outcome == 1:
            return alpha
        if outcome == N-1:
            return epsilon
    if report==1:
        if outcome == 0:
            return epsilon
        if outcome == N-2:
            return beta

    return 0



class AlphaBetaCenter(object):
    """
    Directly compute a payment rule:

    0 | 0 alpha 0  e
    1 | e 0   beta 0
    """
    def __init__(self, world, N=4, alpha_beta=None, epsilon=1):
        self.world = world
        self.N = N
        if alpha_beta is not None:
            self.alpha, self.beta = alpha_beta
        else:
            self.alpha, self.beta = compute_alpha_beta(world, N)
        self.epsilon = epsilon


    def payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer,
        look up the score.
        """
        return _alpha_beta_payment(self.alpha, self.beta,
                                   self.N, report, outcome, self.epsilon)


class AverageShadowCenter(object):
    """
    A mechanism center for a binary world, using the peer shadowing
    mechanism (from Learning the Prior in Minimal Peer Prediction,
    Jens and David, section 4.)
    """
    def __init__(self, world, scoring_rule=qsr, d=0.1, N=4):
        """
        Args:
            world: a MultiSignalWorld, has to actually be binary
            scoring_rule: a multi-signal scoring rule.
            d: the delta by which to shadow the posteriors
            N: The total number of agents. There are N-1 reference peers.
        """
        self.world = world
        self.d = d
        self.scoring_rule = scoring_rule
        self.N = N


    def payment(self, report, outcome):
        """
        Given a report from an agent and the number of 1s among the peer, compute the
        agent's payment. The score is the average of the individual RBTS scores for each
        peer.
        """
        (q0, q1) = signal_prior = [self.world.prob_signal(0), self.world.prob_signal(1)]

        if report == 1:
            shadow_prediction = [q0 - self.d, q1 + self.d]
        else:
            shadow_prediction = [q0 + self.d, q1 - self.d]

        score_vs_1 = self.scoring_rule(shadow_prediction, 1)
        score_vs_0 = self.scoring_rule(shadow_prediction, 0)

        return ((score_vs_0 * (self.N - 1 - outcome) + score_vs_1 * outcome) /
                float(self.N-1))


class AlphaBetaShadowCenter(object):
    """
    A center that computes expected payments using shadowing, then converts them to
    alpha-beta form.
    Complicated equations from mathematica, assuming N=4, basic qsr, delta=0.1.
    """
    def __init__(self, world, N=4):
        """
        Args:
            world: a MultiSignalWorld, has to actually be binary
            scoring_rule: a multi-signal scoring rule.
            d: the delta by which to shadow the posteriors
            N: The total number of agents. There are N-1 reference peers.
        """
        self.world = world
        self.N = N

    def payment(self, report, outcome):
        p1  = self.world.prior[1]
        p10 = self.world.conditional[1][0]
        p11 = self.world.conditional[1][1]

        # Math from mathematica
        min_alpha=(p10-p10**2+p11-4*p10*p11+3*p10**2*p11-p11**2+3*p10*p11**2 -
                   p10**2*p11**2)/(3*p10*p11-3*p10**2*p11-3*p10*p11**2+3*p10**2*p11**2)

        alpha = min_alpha + 1

        min_beta = (-p10+p1*p10+3*p10**2-3*p1*p10**2-3*p10**3+3*p1*p10**3+
                    2*p10**4-2*p1*p10**4-p1*p11+3*p1*p11**2-3*p1*p11**3+
                    2*p1*p11**4+3*p10**2*alpha-3*p1*p10**2*alpha-6*p10**3*alpha+
                    6*p1*p10**3*alpha+3*p10**4*alpha-3*p1*p10**4*alpha+
                    3*p1*p11**2*alpha-6*p1*p11**3*alpha+3*p1*p11**4*alpha)/(
                        3*p10**3-3*p1*p10**3-3*p10**4+3*p1*p10**4+3*p1*p11**3-3*p1*p11**4)

        max_beta = (1-4*p10+4*p1*p10+6*p10**2-6*p1*p10**2-5*p10**3+5*p1*p10**3+
                    2*p10**4-2*p1*p10**4-4*p1*p11+6*p1*p11**2-5*p1*p11**3+2*p1*p11**4-
                    3*p10*alpha+3*p1*p10*alpha+9*p10**2*alpha-9*p1*p10**2*alpha-
                    9*p10**3*alpha+9*p1*p10**3*alpha+3*p10**4*alpha-3*p1*p10**4*alpha-
                    3*p1*p11*alpha+9*p1*p11**2*alpha-9*p1*p11**3*alpha+
                    3*p1*p11**4*alpha)/(-3*p10**2+3*p1*p10**2+6*p10**3-
                                        6*p1*p10**3-3*p10**4+3*p1*p10**4-
                                        3*p1*p11**2+6*p1*p11**3-3*p1*p11**4)

        beta = (min_beta + max_beta) / 2

        return _alpha_beta_payment(alpha, beta, self.N, report, outcome)


class PerturbedRBTSCenter(object):
    """
    A center that computes payments as a function of two parameters, like so:

    {{-1, a, b, 11/10}, {0, 1/3, 2/3, 1}};
    """
    def __init__(self, world, a=10, N=4):
        if N != 4:
            raise ValueError("Don't have formula for N != 4 yet")

        self.world = world
        self.N = N

        # These are fixed #s from mathematica to start--only valid for p1=1/2, q1=3/5,
        # p11 < 0.9
        # a >= 5693/4140 && b == 1/540 (533 - 360 a)

        self.a = a
        self.b = 1.0/540 * (533.0 - 360.0 * a)


    def payment(self, report, outcome):
        a, b = self.a, self.b

        matrix = [[-1, a, b, 11/10.0],
                  [0, 1/3.0, 2/3.0, 1]]
        return matrix[report][outcome]



# OLD STUFF BELOW



def qsr_binary(prediction_high, observed):
    """
    Compute the Quadratic Scoring Rule for binary signals.

    Args:
        prediction_high: float. The predicted probability of the signal being 1 (high).
        observed: int.  The observed outcome--0 or 1.
    """
    if observed == 0:
        return 1.0 - prediction_high**2
    elif observed == 1:
        return 2*prediction_high - prediction_high**2
    else:
        raise ValueError("observed is {}.  Expected 0 or 1".format(observed))


class MRZCenter(object):
    """
    A mechanism center for a world, using the MRZ mechanism.
    """
    def __init__(self, world, scoring_rule, N=2):
        """
        Args:
            world: a BinaryWorld
            scoring_rule: a function that takes (pred_high, actual)
                and returns a float.
            N: optional (default 2).  How many agents play in each round.
               If >2, use more than one reference rater, give average payoff
               accross all of them.
        """
        self.world = world
        self.scoring_rule = scoring_rule
        self.N = N


    def payment(self, report, peer_report):
        """
        Given a report from an agent and a peer_report, compute the
        agent's payment.
        """
        posterior_prob_peer_report_high = self.world.posterior_others_signal(report, 1)
        return self.scoring_rule(posterior_prob_peer_report_high, peer_report)


    def _payment(self, report, num_ones):
        """
        Return the agent's payment, given their report and the count of reference
        raters who report one.

        (TODO: generalize to multi-signal)
        """

        num_zeros = self.N - 1 - num_ones
        return ((self.payment(report, 1) * num_ones +
                self.payment(report, 0) * num_zeros) / (self.N - 1))


    def expected_payment_given_signal_truthful(self, signal):
        """
        Given that I saw a particular signal, what's the expected payment
        for truthful reporting (given others also truthful).
        """
        return sum(self.payment(signal, peer_report) *
                   self.world.posterior_others_signal(signal, peer_report)
                   for peer_report in range(self.world.n))


    def expected_payment_truthful(self):
        """
        Expected payment for truthful eq, before seeing any signal.
        """
        return sum(self.expected_payment_given_signal_truthful(signal) *
                   self.world.prob_signal(signal)
                   for signal in range(self.world.n))




class JF09Center(object):
    """
    Mechanism center for a binary world, using the Jurca Faltings '09 mechanism
    (unique Nash eq, section 4.1), set up with a set of peer reports.

    Normalizes the score so the max is 1.
    """
    def __init__(self, world, N):
        """
        N -- number of raters to eval at a time. (N-1 references for each)
        """
        if N < 4:
            raise ValueError("Can't use JF09 with N < 4. Passed N={}".format(N))
        self.world = world
        self.N = N
        self.epsilon = 0.1
        # define shortcut
        def P(n, obs):
            return world.prob_signal_count(n, N, 1, obs)

        # Define some helpers to do robust floating point comparisons:
        # otherwise symmetric worlds cause problems
        def gt(a, b):
            """
            Greater than with a tolerance for floating point errors.
            """
            return (a-b) > 0.0000001

        def lt(a,b):
            return gt(b,a)

        ineq1 = gt(P(1,0) * P(1,1), P(N-2, 0) * P(N-2, 1))
        ineq2 = lt(P(1,0) * P(1,1), P(N-2, 0) * P(N-2, 1))

        self.condA = (ineq1 and gt(P(N-2, 1), P(1,1)) and
                 gt(P(N-2, 1)**2 - P(1, 1)**2, P(1, 0) * P(1, 1) - P(N-2, 0)* P(N-2, 1)))
        self.condB = (ineq2 and gt(P(1, 0), P(N-2, 0)) and
                 gt(P(1,0)**2 - P(N-2,0)**2, P(N-2,0)*P(N-2,1) - P(1,0)*P(1,1)))

        # Define tau(0,1), tau(1, N-2)
        if self.condA:
            tau01 = P(1,1)/(P(1,0) * P(1,1) - P(N-2,0) * P(N-2,1))
            tau1Nm2 = P(N-2,1)/(P(1,0)*P(1,1) - P(N-2,0) * P(N-2,1))
        elif self.condB:
            tau01 = P(1,0)/(P(N-2,0)*P(N-2,1)-P(1,0)*P(1,1))
            tau1Nm2 = P(N-2,0)/(P(N-2,0)*P(N-2,1)-P(1,0)*P(1,1))
        else:
            tau01 = (P(N-2,1) + P(N-2,0))/(P(1,0)*P(N-2,1) - P(N-2,0)*P(1,1))
            tau1Nm2 = (P(1,1) + P(1,0) )/(P(1,0)*P(N-2,1)-P(N-2,0)*P(1,1))

        self.tau01 = tau01
        self.tau1Nm2 = tau1Nm2

        self.normalization = float(max(self.epsilon, self.tau01, self.tau1Nm2))
        assert min(self.epsilon, self.tau01, self.tau1Nm2) >= 0
        assert self.normalization > 0

    def payment(self, report, num_ones):
        """
        Return the agent's payment, given their report and the count of reference
        raters who report one.
        """
        if report == 0:
            if num_ones != 1 and num_ones != self.N-1:
                return 0
            elif num_ones == self.N-1:
                return self.epsilon / self.normalization
            else: # num_ones = 1
                return self.tau01 / self.normalization
        else:
            if num_ones != 0 and num_ones != self.N-2:
                return 0
            elif num_ones == 0:
                return self.epsilon / self.normalization
            else: # num_ones = self.N-2
                return self.tau1Nm2 / self.normalization


    def payment_for_ref_reports(self, report, reference_reports):
        """
        Given a report from an agent, a list of reference reports, compute
        the agent's payment.
        """
        num_ones = sum(reference_reports)
        return self.payment(report, num_ones)


    def expected_payment_given_signal_truthful(self, signal):
        """
        Given that I saw a particular signal, what's the expected payment
        for truthful reporting (given others also truthful).
        """
        return sum(self._payment(signal, num_ones) *
                   self.world.prob_signal_count(num_ones, self.N, 1, signal)
                   for num_ones in range(self.N))


    def expected_payment_truthful(self):
        """
        Expected payment for truthful eq, before seeing any signal.
        """
        return sum(self.expected_payment_given_signal_truthful(signal) *
                   self.world.prob_signal(signal)
                   for signal in range(self.world.n))


def print_game(center):
    """
    For a binary world, print the center's payments for all pairs of signals.
    """
    for i in range(2):
        print "{}".format(i),
        for j in range(2):
            print "({:5.2} {:5.2}) ".format(center.payment(i,j),
                                            center.payment(j,i)),
        print


def expected_truthful_payment(center):
    """
    Given a world model and a center that defines a payment rule,
    compute the expected payment from truthful reporting.
    """
    expected = 0
    for i in range(2):
        for j in range(2):
            prob = center.world.prob_signal_pair(i, j)
            payment = center.payment(i, j)
            expected += prob * payment

    return expected

def expected_misreporting_payment(center, report):
    """
    Given the world model, a center (payment rule), compute the expected
    payment for the first agent reporting report, assuming the other agent
    is truthful.
    """
    expected = 0
    for s in range(2):
        prob = center.world.prob_signal(s)
        payment = center.payment(report, s)
        expected += prob * payment

    return expected

def print_game_properties(center, name=""):
    if name:
        print name + ":"
    print_game(center)
    print "Expected payment if truthful: {:.3}".format(
        expected_truthful_payment(center))
    print "Expected payment if always report 0, 1: {:.3}, {:.3}".format(
        expected_misreporting_payment(center, 0),
        expected_misreporting_payment(center, 1))
    print

def mrz(world):
    return MRZCenter(world, qsr_binary)

def mrz_gap(world):
    """
    Compute the difference between the best static nash eq (better of 1,1 and 0,0)
    and the expected payment if truthful, for mrz in a particular world.
    """
    center = mrz(world)
    best_static = max(center.payment(1, 1), center.payment(0, 0))
    expected_truthful = expected_truthful_payment(center)
    return best_static - expected_truthful

def shadow(world):
    return ShadowCenter(world, 0.2, qsr_binary)

def shadow_gap(world):
    """
    Compute the difference between the best static nash eq (better of 1,1 and 0,0)
    and the expected payment if truthful, for mrz in a particular world.
    """
    center = shadow(world)
    best_static = max(center.payment(1, 1), center.payment(0, 0))
    expected_truthful = expected_truthful_payment(center)
    return best_static - expected_truthful


# look at populations

def expected_truthful_vs_fixed(center, fixed_report):
    """
    Compute the expected payment for a truthful agent when the matched
    agent always reports fixed_report.
    """
    expected = 0
    for s in range(2):
        prob = center.world.prob_signal(s)
        payment = center.payment(s, fixed_report)
        expected += prob * payment

    return expected

class Population(object):
    def __init__(self, fraction_truthful, focal_report):
        """
        fraction_truthful
        assumes that fraction_uninformed is the rest
        focal_report is the symmetric equilibrium the non-informed
          players are playing.
        """
        self.fraction_truthful = fraction_truthful
        self.fraction_uninformed = 1.0 - fraction_truthful
        self.focal_report = focal_report

    def expected_payment_truthful(self, center):
        """
        Compute the expected payment for the truthful group.
        """
        expected_payment_vs_truthful = expected_truthful_payment(center)
        expected_payment_vs_uninformed = expected_truthful_vs_fixed(
            center, self.focal_report)

        return (self.fraction_truthful * expected_payment_vs_truthful +
         self.fraction_uninformed * expected_payment_vs_uninformed)

    def expected_payment_fixed(self, center):
        """
        Compute the expected payment for the fixed-report group.
        """
        exp_vs_truthful = expected_misreporting_payment(center,
                                                        self.focal_report)
        exp_vs_uninformed = center.payment(self.focal_report,
                                           self.focal_report)
        return (self.fraction_truthful * exp_vs_truthful +
                self.fraction_uninformed * exp_vs_uninformed)




###  Plotting


