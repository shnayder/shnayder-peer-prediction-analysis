"""
Some handy utilities, and a bunch of analysis code, most of which isn't used in the final
version of my dissertation.

Author: Victor Shnayder <shnayder@gmail.com>
"""
def prettyfloats(lst):
    """pretty version of a list of floats"""
    return "["+", ".join("{:6.3f}".format(x) for x in lst)+"]"


def prettyints(lst):
    """
    pretty version of a list of ints
    (lines up with a corresponding list of prettyfloats)
    """
    return "["+", ".join("{:6d}".format(x) for x in lst)+"]"


def scores_for_report(center, report):
    """
    Return a list of scores for a particular report, and different realizations of numbers
    of 1s among the N-1 reference raters.

    Args:
        center: a center that has a payment(report, outcome) method.
        report: 0 or 1 -- what 'I' reported
    """

    return [center.payment(report, outcome) for outcome in range(center.N)]


def probs_1s_given_signal(world, signal, N=4):
    """
    Return a list of likelyhoods, for i in range(N), that i out of N-1 people will get a
    1, given that I saw signal,
    """
    return [world.prob_signal_count(i, N, 1, signal) for i in range(N)]


def outcome_payments_given_signal(center, signal, report):
    """
    Given that I saw a particular signal, what are the contributions to the expected
    payment for reporting report (given others are truthful), for each outcome. The sum of
    these will be the overall expected payment--they are not conditioned on that outcome
    actually happening.

    Example:
        If the payments for outcomes are [1, 2, 3, 4], and the probabilies of those
        outcomes are [0.2, 0.2, 0.3, 0.3], this will return [0.2, 0.4, 0.9, and 1.2].

    Returns:
        a list of center.N floats.
    """
    # probabilities and scores for various numbers of 1s
    # probs based on what I actually saw
    dist = probs_1s_given_signal(center.world, signal, center.N)
    # scores based on what I report
    scores = scores_for_report(center, report)

    return [prob * score for prob, score in zip(dist, scores)]


def expected_payment_given_signal(center, signal, report):
    """
    Given that I saw a particular signal, what's the expected payment
    for reporting report (given others are truthful).
    """
    payments = outcome_payments_given_signal(center, signal, report)
    return sum(payments)


def expected_payment_given_signal_truthful(center, signal):
    """
    Given that I saw a particular signal, what's the expected payment
    for truthful reporting (given others also truthful).
    """
    # honest reporting -- report same as signal
    report = signal

    return expected_payment_given_signal(center, signal, report)


def expected_payment_truthful(center):
    """
    Expected payment for truthful eq, before seeing any signal.
    """
    return sum(expected_payment_given_signal_truthful(center, signal) *
               center.world.prob_signal(signal)
               for signal in range(center.world.n))


def expected_payment_lie(center):
    """
    Expected payment for lie equilibrium: if everyone always lies.
    (two agent, binary only)
    """
    total = 0
    w = center.world
    for my_signal in range(2):
        for other_signal in range(2):
            prob = w.prob_signal(my_signal) * w.posterior_others_signal(
                my_signal, other_signal)

            payment = center.payment(1-my_signal, 1-other_signal)

            total += prob * payment

    return total


def expected_payment_lie_given_truthful(center):
    """
    Expected payment for lie if everyone else is truthful.
    (two agent, binary only)
    """
    total = 0
    w = center.world
    for my_signal in range(2):
        for other_signal in range(2):
            prob = w.prob_signal(my_signal) * w.posterior_others_signal(
                my_signal, other_signal)

            payment = center.payment(1-my_signal, other_signal)

            total += prob * payment

    return total



def payment_others_fixed_reports(center, report, fixed_report):
    """
    Payment (deterministic) if everyone gives the same fixed report, and you
    report report.
    (NOTE: only works for binary)
    """
    # probabilities and scores for various numbers of 1s
    scores = scores_for_report(center, report)
    if fixed_report == 0:
        # Everyone else will then report 0 too
        return scores[0]
    else:
        # everyone reports 1 too
        return scores[-1]


def expected_payment_fixed_given_truthful(center, fixed_report):
    """
    Expected payment for reporting fixed_report given that everyone else
    is truthful.
    """
    return sum(sum(outcome_payments_given_signal(center, signal, fixed_report)) *
               center.world.prob_signal(signal)
               for signal in range(center.world.n))


def payment_all_fixed_reports(center, fixed_report):
    """
    Payment (deterministic) if everyone gives the same fixed report.
    (NOTE: only works for binary)
    """
    return payment_others_fixed_reports(center, fixed_report, fixed_report)


def truth_over_misreport_deltas(center):
    """
    Return a list of deltas, one for each possible signal: the delta is how much better
    truthful reporting is over the best misreport, given truthful reporting by others
    """
    def delta(signal):
        expected_honest_payment = expected_payment_given_signal_truthful(center, signal)
        misreports = [report for report in range(center.world.n)
                      if report != signal]
        expected_misreport_payments = [expected_payment_given_signal(
                                            center, signal, report)
                                       for report in misreports]
        return expected_honest_payment - max(expected_misreport_payments)

    return [delta(signal) for signal in range(center.world.n)]


def is_truthful_equilibrium(center):
    """
    Given a world and a scoring rule, return a bool of whether truthful reporting
    is an equilibrium.
    """
    # if all deltas are positive, no useful misreports
    return all(d > 0 for d in truth_over_misreport_deltas(center))


def is_fixed_report_equilibrium(center, fixed_report):
    """
    Given a world and a scoring rule, return a bool of whether everyone
    reporting fixed_report is an equilibrium.
    """
    expected_honest_payment = payment_all_fixed_reports(center, fixed_report)
    misreports = [report for report in range(center.world.n) if report != fixed_report]
    expected_misreport_payments = [payment_others_fixed_reports(center, report, fixed_report)
                                   for report in misreports]
    if max(expected_misreport_payments) > expected_honest_payment:
        return False

    return True


def constraints_met(center):
    """
    True if we like this setup
    """
    return (is_truthful_equilibrium(center) and
            not any(is_fixed_report_equilibrium(center, i) for i in range(center.world.n)))

def print_expectations_table(center):
    world = center.world
    print "\nScores ({})".format(world.name)
    for report in range(world.n):
        print "{}: {}".format(report, prettyfloats(scores_for_report(center, report)))

    print "\nsignal, report: Expected scores/outcome ({}), expected payment".format(world.name)
    for signal in range(world.n):
        for report in range(world.n):
            scores = outcome_payments_given_signal(center, signal, report)
            print "{}, {}: {}, total: {:6.2f}".format(signal, report,
                                                      prettyfloats(scores), sum(scores))

    print "\nExpected values for truthful eq.({})".format(world.name)
    print "Expected given signal"
    for signal in range(world.n):
        print "{}: {:6.2f}  misreport: {:6.2f}".format(signal,
                              expected_payment_given_signal_truthful(center, signal),
                               # consider only binary worlds, hence 1-signal
                              expected_payment_given_signal(center, signal, 1-signal))
    print "Expected a-priori: {:6.2f}".format(expected_payment_truthful(center))
    print "All-0s payment: {:6.2f}".format(payment_all_fixed_reports(center, 0))
    print "All-1s payment: {:6.2f}".format(payment_all_fixed_reports(center, 1))

    print "\nExpected values for report given others truthful: {}".format(
        prettyfloats([expected_payment_fixed_given_truthful(center, report)
         for report in range(world.n)]))

    truthful = "Truthful" if is_truthful_equilibrium(center) else "Not Truthful"
    all_0s = "All 0s is Eq" if is_fixed_report_equilibrium(center, 0) else "All 0s NOT Eq"
    all_1s = "All 1s is Eq" if is_fixed_report_equilibrium(center, 1) else "All 1s NOT Eq"
    print "({:20}): {:20} {:20} {:20}".format(center.world.name, truthful,
                                              all_0s, all_1s)


def analyze_da_world(center):
    """
    Print a bunch of info about a DevilsAdvocateCenter + world
    """
    # Late import to avoid circular deps
    from scoring import ScoringRuleCenter

    world = center.world

    # We also want to be able to print what will happen without the
    # devil's advocate payments
    base_center = ScoringRuleCenter(world, center.scoring_rule, center.N)

    print "World ({}) ############".format(world.name)
    print world

    print "Probabilities of outcomes given signal:"
    for signal in range(world.n):
        print "{}: {}".format(signal,
            prettyfloats(probs_1s_given_signal(world, signal, center.N)))

    print "\nBase payments"
    for report in range(world.n):
        print "{}: {}".format(report, prettyfloats(scores_for_report(base_center, report)))

    print "\nPayments"
    for report in range(world.n):
        print "{}: {}".format(report, prettyfloats(scores_for_report(center, report)))

    print "\nExpected base payments/outcome: signal, report, payments"
    for signal in range(world.n):
        for report in range(world.n):
            scores = outcome_payments_given_signal(base_center, signal, report)
            print "{}, {}: {}, total: {:6.3f}".format(signal, report,
                                                      prettyfloats(scores), sum(scores))

    print "\nExpected payments/outcome: signal, report, payments"
    for signal in range(world.n):
        for report in range(world.n):
            scores = outcome_payments_given_signal(center, signal, report)
            print "{}, {}: {}, total: {:6.3f}".format(signal, report,
                                                      prettyfloats(scores), sum(scores))

    print "\nExpected values for truthful eq."
    print "Expected given signal"
    for signal in range(world.n):
        print "{}: {:6.3f}  misreport: {:6.3f}".format(signal,
                              expected_payment_given_signal_truthful(center, signal),
                               # consider only binary worlds, hence 1-signal
                              expected_payment_given_signal(center, signal, 1-signal))

    print "\ndeltas:"
    print "0: {:6.3f}   1: {:6.3f}".format(center.delta0, center.delta1)


    print "Expected a-priori: {:6.3f}".format(expected_payment_truthful(center))
    print "All-0s payment: {:6.3f}".format(payment_all_fixed_reports(center, 0))
    print "All-1s payment: {:6.3f}".format(payment_all_fixed_reports(center, 1))

    truthful = "Truthful" if is_truthful_equilibrium(center) else "Not Truthful"
    all_0s = "All 0s is Eq" if is_fixed_report_equilibrium(center, 0) else "All 0s NOT Eq"
    all_1s = "All 1s is Eq" if is_fixed_report_equilibrium(center, 1) else "All 1s NOT Eq"
    print "Equilibria: {:20} {:20} {:20}".format(truthful, all_0s, all_1s)

    print

    print "x: {:6.3f} ok: {}, x_slack: {:6.3f} delta0: {:6.3f} bonus_for_zero: {:6.3f}".format(
        center.x, center.x_ok, center.x_slack, center.delta0, center.bonus_for_zero)
    print "y: {:6.3f} ok: {}, y_slack: {:6.3f} delta1: {:6.3f} bonus_for_N: {:6.3f}".format(
        center.y, center.y_ok, center.y_slack, center.delta1, center.bonus_for_N)

    print "\nConstraints:"
    P = lambda my_signal, count: center.world.prob_signal_count(count, center.N,
                                                                1, my_signal)

    print "x * P(0,0) - y * P(0,3) < delta_0 : {:8.5f} < {:8.5f} => {}".format(
        center.x * P(0,0) - center.y * P(0,3),
        center.delta0,
        center.x * P(0,0) - center.y * P(0,3) < center.delta0)
    print "y * P(1,3) - x * P(1,0) < delta_1 : {:8.5f} < {:8.5f} => {}".format(
        center.y * P(1,3) - center.x * P(1,0),
        center.delta1,
        center.y * P(1,3) - center.x * P(1,0) < center.delta1)


def binary_world_str(world):
    """
    Given a world with two signals, return a one line summary.
    """
    return "P(1): {:4.2} P(1|0): {:4.2} P(1|1): {:4.2}".format(
        world.prior[1],
        world.conditional[1][0],
        world.conditional[1][1],
        )


