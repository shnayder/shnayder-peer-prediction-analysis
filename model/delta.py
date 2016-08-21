"""
Wrangling of delta matrices, joint distributions, etc.

Author: Victor Shnayder <shnayder@gmail.com>
"""
import collections as col
import itertools as it
import math
import numpy as np
import operator as op

import edx_data.edx_data as edx

import world

by_criterion = edx.grouper_by_keys("course_id", "item_id", "criterion_name")


def estimate_signal_joint(submission_lists):
    """
    Args:
        submission_lists: A list of submissions, where each submission is a
        list of 0/1/...max assessments.
        e.g. [[0,3,1], [1,1], [0,1,2,0]]
    Returns:
        np array of an estimated signal joint probability.

    Notes:
        - To avoid over-weighting reports for submissions with more assessments, only use
        one peer for each report. (Impl detail: will use the next one -- (i+1)%n)
        - Ensures joint is symmetric
    """
    # count pairs of signals
    joint_cnts = col.Counter()
    for rs in submission_lists:
        n = len(rs)
        # skip submissions with only one report
        if n < 2:
            continue
        for i, r in enumerate(rs):
            peer = (i + 1) % n
            # for off-diagonal, assign to r < r', and re-distribute later
            r2 = rs[peer]
            idx = (r, r2) if r < r2 else (r2,r)
            joint_cnts[idx] += 1

    # figure out bounds
    max_val = max(it.chain.from_iterable(joint_cnts.keys()))
    min_val = 0  # just assume min is 0 (will be, given adjustment)

    # normalize and convert to array
    total = float(sum(joint_cnts.values()))

    k = max_val + 1
    joint_dist = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i==j:
                joint_dist[i,i] = joint_cnts[(i,i)] / total
            elif i<j:
                # leave half the weight for j,i
                joint_dist[i,j] = joint_cnts[(i,j)] / (2*total)
            else:
                # for i > j, need to look in j,i
                joint_dist[i,j] = joint_cnts[(j,i)] / (2*total)
    return joint_dist


def prior_from_joint(joint):
    """
    Given a NxN joint array, returns an array of len N with the prior.

    Assumes joint is symmetric.
    """
    return np.array(joint.sum(0).flat) # assume symmetry


def normalize(x):
    return x / float(np.sum(x))


def independent_joint(xs, ys):
    """
    Given two distributions, return a joint distribution assuming they're independent.

    Args:
        xs: np.array of length n
        ys: np.array of length m

    Returns:
        np.array(size=(n,m)), element i,j is xs[i]*ys[j]
    """
    # not the most efficient, but doesn't matter
    result = np.zeros((len(xs), len(ys)))
    for i in range(len(xs)):
        for j in range(len(ys)):
            result[i,j] = xs[i] * ys[j]

    return result


def ind_joint_from_joint(joint):
    dir1 = joint.sum(1).flatten()
    dir2 = joint.sum(0).flatten()
    ind_joint = independent_joint(dir1, dir2)
    return ind_joint


def delta_from_joint(joint):
    """
    joint is an n by n matrix. Does not have to be symmetric.
    """
    joint = np.array(joint)
    ind_joint = ind_joint_from_joint(joint)
    return joint - ind_joint


def flatten(iter_of_iters):
    "Flatten one level of nesting"
    return it.chain.from_iterable(iter_of_iters)


# Double checking estimation function above.
def sample_signal_joint(submission_lists, k=1000):
    """
    Compute sample of signal joint.

    Args:
        submission_lists: A list of submissions, where each submission is a
        list of 0/1/...max assessments.
        e.g. [[0,3,1], [1,1], [0,1,2,0]]
    Returns:
        np array of an estimated signal joint probability, based on k samples.

    Notes:
        - each sample is taken by choosing a submission with at least two assessments
         (with replacement), then taking two random different assessments from that
         submission.
        - Ensures joint is symmetric
    """
        # count pairs of signals
    joint_cnts = col.Counter()

    eligible = [xs for xs in submission_lists if len(xs) > 1]
    weights = map(len, eligible)
    total = sum(weights)
    probs = [float(w) / total for w in weights]

    for i in np.random.choice(len(eligible), k, replace=True, p=probs):
        r, r2 = np.random.choice(eligible[i], size=2, replace=False)
        # for off-diagonal, assign to r < r', and re-distribute later
        idx = (r, r2) if r < r2 else (r2,r)
        joint_cnts[idx] += 1

    # figure out bounds
    max_val = max(flatten(joint_cnts.keys()))
    min_val = 0  # just assume min is 0 (will be given adjustment)

    # normalize and convert to array
    total = float(sum(joint_cnts.values()))

    k = max_val + 1
    joint_dist = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i==j:
                joint_dist[i,i] = joint_cnts[(i,i)] / total
            elif i<j:
                # leave half the weight for j,i
                joint_dist[i,j] = joint_cnts[(i,j)] / (2*total)
            else:
                # for i > j, need to look in j,i
                joint_dist[i,j] = joint_cnts[(j,i)] / (2*total)
    return joint_dist



#### Conditions on joint/delta

def offdiag(matrix):
    """
    Return an array of all off-diagonal elements as a 1D array.
    """
    n = matrix.shape[0]
    return np.extract(1 -  np.eye(n), matrix)


def negative_off_diag(delta):
    return (offdiag(delta) < 0).all()


def positive_diag(delta):
    return (np.diag(delta) > 0).all()


def is_categorical(delta):
    """
    Given a square array delta, returns True if it has positive diagonal elements,
    negative off-diagonal ones.
    """
    # assumes square
    return positive_diag(delta) and negative_off_diag(delta)

def is_self_predicting(joint):
    """
    Given a joint, returns True if it's self-predicting:
        P(other_signal = x|my signal = x) > P(other_signal = x|my signal = y)
            for all x, y
            (shorter: P(s|s) > P(s|s'))
    """
    return world.is_self_predicting(joint)


def is_self_dominant(joint):
    """
    Given a joint, returns True if it's self-dominant:
        P(other_signal = x|my signal = x) > P(other_signal = y|my signal = x)
            for all x, y
            (shorter: P(s|s) > P(s'|s))
    """
    prior = np.array(joint.sum(0).flat) # assume symmetry

    def post(s_given_peer, s_me):
        """Posterior prob of another agent's signal given mine"""
        #        print joint, prior, s_me, s_given_peer
        return joint[s_me, s_given_peer] / prior[s_me]

    n = joint.shape[0]
    for x in range(n):
        for y in range(x+1, n): # assumes symmetry again
            if post(y,x) > post(x,x):
                return False

    return True

def is_shadowing_safe(joint):
    """
    Given a joint, returns True if it satisfies the additive peer shadowing condition:
            P(s|s) - P(s) > P(s'|s) - P(s')
            for all s != s'
    """
    prior = np.array(joint.sum(0).flat) # assume symmetry

    def post(s_given_peer, s_me):
        """Posterior prob of another agent's signal given mine"""
        #        print joint, prior, s_me, s_given_peer
        return joint[s_me, s_given_peer] / prior[s_me]

    n = joint.shape[0]
    for x in range(n):
        for y in range(x+1,n):
            if post(x,x) - prior[x] < post(y,x) - prior[y]:
                return False

    return True

#### Scoring algorithms -- similar to scorers in dynamics.py, but
# data structs are sufficiently different it's easier to rewrite.

# def _output_agreement_score(reports, avg):
#     """
#     Helper to actually do the work.
#     """
#     n = len(reports)
#     if n == 1:
#         return [0.5]

#     feedback = np.zeros(n)
#     counts = col.Counter(reports)
#     for i, report in enumerate(reports):
#         if not avg:
#             peer = (i+1) % n
#             feedback[i] = int(reports[peer] == report)
#         else:
#             # avg agreement with all peers
#             # (subtract 1 to not count self-agreement)
#             feedback[i] = (counts[report]-1)/float(n-1)

#     return feedback

# def output_agreement_scores(submissions, avg=True):
#     """
#     Args:
#         submissions: output of edx.group_scores_by_submission
#         avg: if True, avg agreement with all peers

#     Returns:
#         list of lists of scores. 0 or 1, or 0.5 if there's only one report for a
#         submission.
#     """
#     return [_output_agreement_score(reports, avg)
#             for reports in submissions]



def scores_oa(assessments, avg=False):
    """
    Implement the OA mechanism on assessments for a particular criterion.

    Args:
        assessments: an iterator, to be processed once (e.g. output of groupby),
          of assessments for a particular criterion.

    Returns:
        list of scores, one for each assessment.

    Algorithm:
    - compute report frequencies
    - for each assessment, pick random peer. If match, score = 1 frequency. If not
      match, score = 0.
    """
    # We'll need to iterate through multiple times, so save as a list
    assessments = list(assessments)

    # To pick a peer or average over peers, need to know who they are.
    # Compute the assessments (by index) for each submission
    # submission_uuid -> list(indexes into assessments)
    graded_by = col.defaultdict(list)

    for i, a in enumerate(assessments):
        graded_by[a.submission_uuid].append(i)

    # Ok, now we're ready to compute scores

    def get_peers(i):
        """Return list of peer indexes for assessment with index i. Respect avg."""
        a = assessments[i]
        peers = graded_by[a.submission_uuid]
        if avg:
            # everyone but self
            return set(peers) - set([i])

        # otherwise, pick the next peer, wrapping around. Linear search, but who cares.
        k = peers.index(i)
        return [peers[(k+1) % len(peers)]]

    def score(i):
        """Compute score for assessment with index i"""
        a = assessments[i]
        peers = get_peers(i)
        if len(peers) == 0:
            return 0.5
        return np.mean([int(a.points == assessments[j].points)
                        for j in peers])

    return [score(i) for i in range(len(assessments))]

class NoPeerException(Exception):
    pass


def trim_delta(delta, eps=0.01):
    """
    Replace small (abs value < eps) entries of delta with 0. Returns new delta.
    """
    return np.where(np.abs(delta) > eps, delta, np.zeros_like(delta))


def expected_01_score_joint(joint, score_matrix=None):
    """
    Sum of the positive entries of the delta that corresponds to joint,
     where score_matrix is positive
    (defaults to sign(delta)), adjusted to make raw scores fit
    in [0,1]
    """
    return expected_01_score(delta_from_joint(joint), score_matrix)

def expected_01_score(delta, score_matrix=None):
    """
    Sum of the positive entries, where score_matrix is positive
    (defaults to sign(delta)), adjusted to make raw scores fit
    in [0,1]
    """
    if score_matrix is None:
        score_matrix = delta
    return 0.5 + 0.5*np.sum(np.where(score_matrix > 0,
                           delta, np.zeros_like(delta)))

def expected_rpts_score(joint):
    """
    sum prior(s) * clamp(1/prior(s)) * P(match signal|signal) =
     = sum P(signal, signal) / clamp(P(signal))
    So very similar to 01 -- divide, rather than subtract.

    Factors in the clamping of 1/prior to min 0.1, and the division by 10.
    """
    prior = prior_from_joint(joint)
    return np.sum(joint[i,i] / max(p,0.1)
        for i,p in enumerate(prior)) / 10.0

def expected_kamble_score(joint):
    """
    sum_s P(s) 1/sqrt(P(s,s)) P(s|s)
      = sum_s P(s) 1/sqrt(P(s,s)) P(s,s)/P(s)
      = sum_s 1/sqrt(P(s,s)) P(s,s)
    """
    return np.sum(joint[i,i] /
                  max(math.sqrt(joint[i,i]), 0.25)
                  for i in range(joint.shape[0])
                  ) / 4.0

def score_matrix_01(delta):
    """
    1 where delta > 0, 0 elsewhere
    """
    if not isinstance(delta, np.ndarray):
        raise TypeError("Expect delta to be an np.ndarray")
    return np.where(delta > 0, np.ones_like(delta), np.zeros_like(delta))


def fixed_score_matrix(delta):
    """
    Return a score matrix that only depends on the size of delta, for experiments.
    For size 2,3, identity
    For size 4+, include one sub and superdiagonal.
    """
    n = delta.shape[0]
    if n < 4:
        return np.identity(n)

    ones = [1]*(n-1)
    return np.identity(n) + np.diag(ones, -1) + np.diag(ones, 1)


def score_kong(r1_s, r2_s, r1s, r2s, score_matrix):
    """
    Compute a sampled Kong-Schoenebeck score, based on the reports on
    shared/non-shared tasks.
    Args:
        r1_s,r2_s: the two agents' shared reports on the same task. (ints)
        r1s, r2s: the two agents' non-shared reports (lists)

    Returns:
        a float score in [-2,2]
    """
    # We want to sample from the distribution:
    # sum_ij (joint_ij - prod_marginal_ij) * expected_sign_ij
    #
    # The joint is a point distribution at r1_s,r2_s.
    # compute the marginals:
    cts1 = col.Counter(r1s)
    cts2 = col.Counter(r2s)
    n1 = float(sum(cts1.values()))
    n2 = float(sum(cts2.values()))

    k = score_matrix.shape[0]
    # and here we are: sum (joint - marginal) * sign(joint-marginal)
    return sum(((1 if (i,j) == (r1_s,r2_s) else 0) -
                cts1[i]/n1 * cts2[j]/n2) * (1 if score_matrix[i,j] > 0 else -1)
                for i in range(k)
                for j in range(k))


def scores_01(assessments, avg=False, fixed_scores=False, kong=False):
    """
    Implement the 01 mechanism on assessments for a particular criterion.

    Args:
        assessments: an iterator, to be processed once (e.g. output of groupby),
          of assessments for a particular criterion.
        avg: if True, average scores for all availabe peers
        fixed_scores: if True, use fixed_score_matrix() rather than defining the
            score based on delta.
        kong: if True, use the Kong-Schoenebeck MI scoring rule rather than 01, still with
            the true or fixed 01 score matrix (aka sign structure).

    Returns:
        list of scores, one for each assessment (in order).

    Notes:
    - Computes the joint and corresponding score matrix from the assessments.
    - Students who don't have a reference peer get score 0.5.
    - Returns scores in [0,1]
    """
    # For each assessment a1, need to find a reference peer, who assessed the same
    # submission and another submission.
    # Also need another submission assessed by the assesor of a1. If a student only
    # assessed one submission, they get score 0.5.
    # Algorithm:
    # - Compute a map reports: (submission_uuid, scorer_id) -> report
    # - Compute map of who scored each submission:
    #                  graded_by: submission_id -> set(user_id)
    # - Compute map of what submissions each agent scored:
    #                  submissions_graded: scorer_id -> set(submission_uuid)
    # - For each assessment i of submission s:
    #      If this grader only assessed submission s, score 0.5.
    #      Else let A be the set of other submissions they assessed.
    #      For all other graders who graded s (in random order), check if they graded
    #          other submissions B, with either:
    #                 |B| > 1 OR B != A
    #      If we can't find one, score 0.5.
    #      If we find one, pick reference submission from A, B, ensuring they are
    #          different.
    #      Score based on the reports on the shared and reference submissions.

    # (submission_uuid, scorer_id) -> score
    reports = col.defaultdict(int)

    # scorer_id -> set(submission_uuid)
    submissions_graded = col.defaultdict(set)

    # submission_uuid -> set(scorer_id)
    graded_by = col.defaultdict(set)

    saved = []
    for a in assessments:
        reports[(a.submission_uuid, a.scorer_id)] = a.points
        submissions_graded[a.scorer_id].add(a.submission_uuid)
        graded_by[a.submission_uuid].add(a.scorer_id)
        # Save the relevant assessment info so we can go through again
        # in same order.
        saved.append((a.submission_uuid, a.scorer_id))

    # print reports
    # print submissions_graded
    # print graded_by

    # list of lists of scores
    submissions = [[reports[sub_id, scorer_id]
                    for scorer_id in graded_by[sub_id]]
                   for sub_id in graded_by.keys()]
    delta = delta_from_joint(estimate_signal_joint(submissions))
    if fixed_scores:
        score_matrix = fixed_score_matrix(delta)
    else:
        score_matrix = score_matrix_01(delta)

    # for final output
    n = len(saved)
    scores = np.zeros(n)


    def score_tasks(scorer_id, sub_id, peer_id, me_subs, you_subs, avg):
        """
        Compute the 01 score.
        Args: my id, shared submission id, peer id, "my" sub_ids, your sub_ids
        """
        # shared reports
        r1_s = reports[(sub_id, scorer_id)]
        r2_s = reports[(sub_id, peer_id)]

        if avg:
            # average all pairs of non-shared assessments
            r1s = [reports[r,scorer_id] for r in me_subs]
            r2s = [reports[r,peer_id] for r in you_subs]
        else:
            # Pick a just_me task and a just_you task ("ns" = "not shared")
            r1s = [reports[(np.random.choice(tuple(me_subs)), scorer_id)]]
            r2s = [reports[(np.random.choice(tuple(you_subs)), peer_id)]]

        if kong:
            base_score = score_kong(r1_s, r2_s, r1s, r2s, score_matrix)
            # [-2,2] -> [0,1]
            return (base_score + 2) / 4.0
        else:
            pairs = [score_matrix[r1_ns, r2_ns]
                     for r1_ns in r1s
                     for r2_ns in r2s]
            base_score = score_matrix[r1_s, r2_s] - np.mean(pairs)
            # scale score from [-1,1] to [0,1]
            return (base_score + 1)/2.0


    def find_peer(sub_id, scorer_id):
        """
        Returns:
            peer_id, me_subs, peer_subs:
                self-explanatory,
                set of submission_ids to use as "just my" reports
                set of submission_ids to use as "just peer" reports
            if there is no appropriate peer, raise NoPeerException()
        """
        # is there at least one "other" task each for this scorer and a peer?
        # my other tasks
        sub1 = submissions_graded[scorer_id] - set([sub_id])
        if len(sub1) == 0:
            # give up
            raise NoPeerException("No other tasks")

        # candidates: everyone else who reported on this sub_id
        candidate_ids = list(graded_by[sub_id] - set([scorer_id]))
        np.random.shuffle(candidate_ids)
        for cand_id in candidate_ids:
            # candidate peer's other tasks
            sub2 = submissions_graded[cand_id] - set([sub_id])
            if len(sub2) == 0 or len(sub1.union(sub2)) < 2:
                # this candidate won't work
                continue
            # ok, it'll work. To divide, start with smallest subset
            if len(sub1) < len(sub2):
                me_subs = sub1
                peer_subs = sub2 - sub1
            else:
                peer_subs = sub2
                me_subs = sub1 - sub2

            # Note that the else case includes case when sub1=sub2, with size > 2.
            # Handle that.
            if len(me_subs) == 0:
                me_subs.add(peer_subs.pop())

            # If we made it this far, should be good
            return cand_id, me_subs, peer_subs

        # but if we make it this far, haven't found anything
        raise NoPeerException("No candidate worked")

    for i, (sub_id, scorer_id) in enumerate(saved):
        try:
            peer_id, me_subs, peer_subs = find_peer(sub_id, scorer_id)
            scores[i] = score_tasks(scorer_id, sub_id, peer_id, me_subs, peer_subs, avg)
        except NoPeerException as e:
            msg = e.args[0]
            if msg == "No other tasks":
                scores[i] = 0.4
            else:
                scores[i] = 0.6

    return scores

def scores_rpts(assessments, avg=False):
    """
    Implement the RPTS mechanism on assessments for a particular criterion.

    Args:
        assessments: an iterator, to be processed once (e.g. output of groupby),
          of assessments for a particular criterion.

    Returns:
        list of scores, one for each assessment.

    Algorithm:
    - compute report frequencies
    - for each assessment, pick random peer. If match, score = 1/report frequency. If not
      match, score = 0.
    """
    # We'll need to iterate through multiple times, so save as a list
    assessments = list(assessments)

    # To pick a peer or average over peers, need to know who they are.
    # Compute the assessments (by index) for each submission
    # submission_uuid -> list(indexes into assessments)
    graded_by = col.defaultdict(list)

    counts = col.Counter()
    for i, a in enumerate(assessments):
        graded_by[a.submission_uuid].append(i)
        # while we're going through, also count report frequencies
        # (Note that paper wants sampling of one report per task to avoid bias,
        # but I'd rather do something clearly deterministic -- over a large population
        # with i.i.d. tasks, should be equivalent)
        counts[a.points] += 1

    n = sum(counts.values())
    # Ok, now we're ready to compute scores

    def get_peers(i):
        """Return list of peer indexes for assessment with index i. Respect avg."""
        a = assessments[i]
        peers = graded_by[a.submission_uuid]
        if avg:
            # everyone but self
            return set(peers) - set([i])

        # otherwise, pick the next peer, wrapping around. Linear search, but who cares.
        k = peers.index(i)
        return [peers[(k+1) % len(peers)]]

    def score(i):
        """Compute score for assessment with index i"""
        a = assessments[i]
        peers = get_peers(i)
        if len(peers) == 0:
            return None
        # clamp 1/prior above by 10 (see note in 2016-03-ORA2-scoring)
        return np.mean([(min(10, float(n)/counts[a.points])
                         if a.points == assessments[j].points else 0)
                        for j in peers])

    raw = [score(i) for i in range(n)]
    m = 10 # divide by 10 to ensure score in [0,1]
    return [s / m if s is not None else 0.5
            for s in raw]


def scores_kamble(assessments, avg=False):
    """
    Implement the Kamble mechanism on assessments for a particular criterion.

    Args:
        assessments: an iterator, to be processed once (e.g. output of groupby),
          of assessments for a particular criterion.

    Returns:
        list of scores, one for each assessment.

    Algorithm:
    - compute report frequencies
    - for each assessment, pick random peer. If match, score = 1/report frequency. If not
      match, score = 0.
    """
    # We'll need to iterate through multiple times, so save as a list
    assessments = list(assessments)

    # To pick a peer or average over peers, need to know who they are.
    # Compute the assessments (by index) for each submission
    # submission_uuid -> list(indexes into assessments)
    graded_by = col.defaultdict(list)

    counts = col.Counter()
    for i, a in enumerate(assessments):
        graded_by[a.submission_uuid].append(i)
        # while we're going through, also count report frequencies
        # (Note that paper wants sampling of one report per task to avoid bias,
        # but I'd rather do something clearly deterministic -- over a large population
        # with i.i.d. tasks, should be equivalent)
        counts[a.points] += 1


    n = sum(counts.values())

    # list of lists of scores
    submissions = [[assessments[i].points
                    for i in graded_by[sub_id]]
                   for sub_id in graded_by.keys()]
    joint = estimate_signal_joint(submissions)

    cap = 4.0

    # Now compute \bar{f}(s_k) in paper notation: adjusted_match_prob(s_k)
    bar_f = {}   # score -> val
    for s in set(flatten(submissions)):  # only look at used score values
        # Cap below at 1/cap
        bar_f[s] = max(math.sqrt(joint[s,s]), 1/cap)

    # Ok, now we're ready to compute scores

    def get_peers(i):
        """Return list of peer indexes for assessment with index i. Respect avg."""
        a = assessments[i]
        peers = graded_by[a.submission_uuid]
        if avg:
            # everyone but self
            return set(peers) - set([i])

        # otherwise, pick the next peer, wrapping around. Linear search, but who cares.
        k = peers.index(i)
        return [peers[(k+1) % len(peers)]]

    def score(i):
        """Compute score for assessment with index i"""
        a = assessments[i]
        peers = get_peers(i)
        if len(peers) == 0:
            return None

        return np.mean([((1/bar_f[a.points] if bar_f[a.points] > 0 else 0)
                         if a.points == assessments[j].points else 0)
                        for j in peers])

    raw = [score(i) for i in range(n)]
    # divide by cap to ensure score in [0,1]
    return [s / cap if s is not None else 1/cap
            for s in raw]



#### Scoring algorithm analysis

def scores_by_criterion(assessments, scoring_fn):
    """
    Args:
        assessments -- must be sorted by criterion
        scoring_fn -- fn: assessments for criterion -> scores
    Returns:
        dict: criterion -> {score -> count}
    """
    scores = dict()
    for i, (k, xs) in enumerate(it.groupby(assessments, key=by_criterion)):
        xs = tuple(xs)
        counts = col.Counter(scoring_fn(xs))
        scores[k] = counts
    return scores

def scores_by_size(assessments, scoring_fn):
    """
    Args:
        assessments -- must be sorted by criterion
        scoring_fn -- fn: assessments for criterion -> scores
    Returns:
        dict: size -> {score -> count}
    """
    score_counts = col.defaultdict(col.Counter)  # size -> {score -> count}
    for i, (k, xs) in enumerate(it.groupby(assessments, key=by_criterion)):
        xs = tuple(xs)
        # print '.',
        # if i%50 == 0:
        #     print ''
        submissions = edx.group_scores_by_submission(xs)
        size = max(md.flatten(submissions))+1
        counts = col.Counter(scoring_fn(xs))
        score_counts[size] += counts
    return score_counts

def scores_given_signal_by_size(assessments, scoring_fn):
    """
    Args:
        assessments -- must be sorted by criterion
        scoring_fn -- fn: assessments for criterion -> scores
            (must return scores in same order as input)
    Returns:
        dict: size -> {report -> {score -> count}}
    """
    # size -> {report->{score -> count}}

    score_counts = col.defaultdict(lambda: col.defaultdict(col.Counter))
    for i, (k, xs) in enumerate(it.groupby(assessments, key=by_criterion)):
        # we'll need to use this multiple times
        xs = tuple(xs)

        # First, figure out the mapping from original reports to adjusted reports
        # (0,1,2...)
        report_counts = col.Counter(a.points for a in xs)
        adjustment = edx.points_adjustment(report_counts)
        # how many signals there are
        size = adjustment[max(adjustment.keys())]+1

        scores = scoring_fn(xs)

        for report, ys in it.groupby(sorted(
                # make (report, score) tuples...
                zip([x.points for x in xs], scores)), key=op.itemgetter(0)):
            counts = col.Counter(map(op.itemgetter(1), ys))
            # don't forget to map to the canonical signal values
            score_counts[size][adjustment[report]] += counts
    return score_counts

#### Counter math (sum, mean, stddev, etc)

def counter_sum(counters):
    """
    Add a bunch of Counters.
    """
    return reduce(lambda x,y: x+y, counters)


def counter_mean(counter):
    """
    Counts must be non-negative, and counter must have len > 0.
    Uses floating point division.
    """
    sum_of_numbers = sum(number*count for number, count in counter.items())
    count = sum(counter.values())
    return sum_of_numbers / float(count)


def counter_stddev(counter):
    """
    Return std dev of a summarized set of numbers, passed in as a counter:

    e.g. {0: 10, 1:10} -> 0.5
    """
    # (from http://stackoverflow.com/questions/33695220/calculate-mean-on-values-in-python-collections-counter)
    sum_of_numbers = sum(number*count for number, count in counter.items())
    count = sum(counter.values())
    mean = sum_of_numbers / float(count)

    total_squares = sum(number*number * count for number, count in counter.items())
    mean_of_squares = total_squares / float(count)
    variance = mean_of_squares - mean * mean
    std_dev = math.sqrt(variance)
    return std_dev


def counter_coeff_var(counter):
    """stddev / mean"""
    return counter_stddev(counter) / counter_mean(counter)


def expected_stddev(scores_given_signal):
    """
    What's the expected standard deviation of the scores.

    Args:
        scores_given_signal: {size -> {report -> {score -> count}}}
             (e.g. output of scores_given_signal_by_size)
    Returns:
        the expected standard deviation, weighing each (size,report) pair
        by the number of scores.

    Example:
        input: {3: {0: {0: 10, 1: 10},   # std = 0.5, count = 20
                    1: {0: 10, 0.5: 20, 1:10},   # std=.353, count = 40
                    2: {0: 20, 1: 10}}}     # std = 0.47, count = 30
        output = (0.5 * 20 + 0.353 * 40 + 0.47 * 30)/90 = 0.424
    """
    # flatten the input into a list of counters
    dists = list(flatten(d.values() for d in scores_given_signal.values()))
    # not sure why, but seems like we can have empty counters sometimes. Filter.
    dists = [d for d in dists
             if len(d) > 0]
    total = float(sum([sum(counts.values()) for counts in dists]))

    # [(weight, stddev)]
    pairs = [(sum(counts.values())/total, counter_stddev(counts))
             for counts in dists]
    return sum(w * s for w,s in pairs)


def expected_mean(scores_given_signal):
    """
    What's the expected mean of the scores.

    Args:
        scores_given_signal: {size -> {report -> {score -> count}}}
             (e.g. output of scores_given_signal_by_size)
    Returns:
        the expected mean score.

    Example:
        input: {3: {0: {0: 10, 1: 10},
                    1: {0: 10, 0.5: 20, 1:10},
                    2: {0: 20, 1: 10}}}
        output = (0 * 40 + 0.5*20 + 1*30) / 90 = 4/9
    """
    # add up all the counters
    total_counts = col.Counter()
    for c in flatten(d.values() for d in scores_given_signal.values()):
        total_counts += c

    return counter_mean(total_counts)


def expected_coeff_var(scores_given_signal):
    """
    What's the expected coefficient of variation of the scores.

    Args:
        scores_given_signal: {size -> {report -> {score -> count}}}
             (e.g. output of scores_given_signal_by_size)
    Returns:
        the expected coeff of var: stdev/mean
    """
    return expected_stddev(scores_given_signal) / expected_mean(scores_given_signal)


####

def _collusion_pairwise_score(joint, strat, other_strat, mech, correlated_prior):
    """
    Return the expected score for a pair of strategies.
    """
    if mech != "01":
        raise ValueError("invalid mech {}".format(mech))

    if strat == "random" or other_strat == "random":
        # anyone facing random gets no reward (aka 0.5 in [0,1])
        return 0.5

    if strat != other_strat:
        # truthful vs correlated -- won't actually be correlated at all, score 0,
        # (aka 0.5 in [0,1])
        return 0.5

    # ok, so both truthful or both correlated
    if strat == "truthful":
        return expected_01_score(delta_from_joint(joint))

    if strat == "correlated":
        # if both agents correlated, score depends on correlated_prior:
        # joint is identity, ind joint comes from correlated prior.
        corr_joint = np.diag(correlated_prior)
        return expected_01_score(delta_from_joint(corr_joint))


def _collusion_expected_score(i, joint, strats, proportions, mech, correlated_prior):
    """
    Compute expected score for a single strategy i.

    Args:
        i: the index of the strategy being considered.
        for rest, see docstring for collusion_exp_scores.
    Returns:
        float.
    """
    score = 0.0
    strat = strats[i]
    for j, other_strat in enumerate(strats):
        score += (proportions[j] *
                  _collusion_pairwise_score(joint, strat, other_strat,
                                            mech, correlated_prior))

    return score

def collusion_exp_scores(joint, strats, proportions, mech, correlated_prior=None):
    """
    Args:
        joint: world joint distribution
        strats: list, with possible values, each occuring at most once:
                ["truthful", "random", "correlated"]
        proportions: list of length len(strats) non-negative floats, with proportions
                of population playing corresponding strategy in strats. Must sum to 1.0.
        mech: mechanism name. Possible values:
                ["O1"].
                Note that resulting 01 scores are unnormalized, in [-1,1]
        correlated_prior: if None, uses uniform prior for correlated strategy. Otherwise,
                uses this (must have length matching joint)
    Returns:
        a list of length len(strats), with the expected payoff for each.
    """
    if mech not in set(["01"]):
        raise ValueError("Invalid mechanism {}".format(mech))

    n = joint.shape[0]
    if correlated_prior is None:
        correlated_prior = [1.0 / n] * n

    return [_collusion_expected_score(i, joint, strats, proportions,
                                      mech, correlated_prior)
            for i in range(len(strats))]

def collusion_indifference_point(joint, correlated_prior=None):
    """
    Return the fraction truthful where agents are indifferent between being
    truthful and perfectly correlated, with the given correlated prior.
    """
    n = joint.shape[0]
    if n == 1:
        raise ValueError("n=1")
    if correlated_prior is None:
        correlated_prior = [1.0 / n] * n

    truthful_score = _collusion_pairwise_score(joint, "truthful", "truthful",
                                               "01", correlated_prior)
    correlated_score = _collusion_pairwise_score(joint, "correlated", "correlated",
                                                 "01", correlated_prior)

    # And now, the math is pretty easy, but need to compensate for "0"
    # actually being 0.5
    # p * truthful_score = (1-p) correlated_score
    # p T = C - p C
    # p (T + C) = C
    # p = C / (T + C)
    return (correlated_score-0.5) / float(truthful_score + correlated_score - 1)
