#!/usr/bin/env python

"""
Loading and cleaning csvs of individual peer assessments.

Author: Victor Shnayder <shnayder@gmail.com>
"""

import csv
import getopt
import gzip
import numpy as np
import sys

# Global code is weird, but seems to be the right way
csv.register_dialect('MyDialect', delimiter=',',
                     doublequote=False, quotechar='"', lineterminator='\n',
                     escapechar='\\',strict=True)

from datetime import datetime
from operator import itemgetter
from itertools import groupby, imap, izip, count, chain
from collections import namedtuple, Counter, deque, Iterable

HEADERS = ["course_id", "item_id", "student_id", "submission_uuid",
                        "score_type", "scorer_id", "scored_at", "criterion_name",
                        "option_name", "points", "len_criterion_feedback", "len_overall_feedback"]

SIMPLIFY = set(["item_id", "student_id", "submission_uuid", "scorer_id"])

Assessment = namedtuple("Assessment", HEADERS)


def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = count()
    deque(izip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def unique_justseen(iterable, key=None):
    """
    From the itertools page.

    List unique elements, preserving order. Remember only the element just seen.
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    """
    return imap(next, imap(itemgetter(1), groupby(iterable, key)))


def load_assessments(filename, assume_clean=False):
    """
    Load assessments from a csv file with columns

    "course_id", "item_id", "student_id", "submission_uuid",
    "score_type", "scorer_id", "scored_at", "criterion_name",
    "option_name", "points", "criterion_feedback", "overall_feedback"

    De-duplicate the list.

    Convert points to an int
    convert scored_at to a datetime

    If filename ends in gz, unzips it

    Returns: list of assessments
    """
    if filename.endswith('.gz'):
        f = gzip.GzipFile(filename, 'rb')
    else:
        f = open(filename)

    data = []
    reader = csv.reader(f, dialect='MyDialect')
    # Skip the header row
    reader.next()
    skips = []
    for row in reader:
        try:
            elt = Assessment(*row)
            elt = elt._replace(points=int(elt.points),
                               scored_at=datetime.strptime(elt.scored_at,
                                                           '%Y-%m-%d %H:%M:%S'))
            data.append(elt)
        except ValueError:
            # Skip rows that don't have an int for points, or improper time
            skips.append(reader.line_num)
            continue

    f.close()

    if not assume_clean:
        # this can take a bit of time on large datasets, so we have an option to skip it
        # if it's already been done
        if skips:
            print "Skipped some rows. Line numbers: {}".format(skips)
        data.sort()
        return list(unique_justseen(data))
    else:
        return data

def save_assessments(filename, assessments):
    """
    Save assessments to filename, zipping if filename ends in '.gz'
    """
    if filename.endswith('.gz'):
        f = gzip.GzipFile(filename, 'wb')
    else:
        f = open(filename, 'wb')
    writer = csv.writer(f, dialect='MyDialect')

    writer.writerow(HEADERS)
    for r in assessments:
        writer.writerow(r)
    f.close()


def grouper_by_keys(*args):
    """
    Returns a key function that will take a namedtuple and return a tuple with
    the values for the keys passed in.
    """
    def f(d):
        return tuple(getattr(d, x) for x in args)

    return f


def keep_latest_scores(assessments):
    """
    Given a list of assessments, keep only the latest scores.

    Keeps the latest score for each (submission, scorer, criterion) tuple.
    """
    key_fn = grouper_by_keys('submission_uuid','scorer_id','criterion_name')
    groups = [list(group) for k, group in groupby(sorted(assessments, key=key_fn),
                                                  key_fn)]
    return [max(group, key=lambda x: x.scored_at)
            for group in groups]


def restrict_to_score_types(assessments, types):
    """
    Return just the assessments that have a score type type in types.

    Args:
        assessments: list
        types: either a single value (e.g. "PE") or a sequence (e.g. ("PE", "SE")).
    """
    if not isinstance(types, Iterable):
        types = (types, )

    return [a for a in assessments
            if a.score_type in types]


def convert_to_binary(criterion_assessments):
    """
    Given a set of assessments for a single criterion, group them by submission and
    convert them to 0/1 values.

    Discretizes based on max score--1 for full credit, 0 otherwise.

    Uses the max score among the passed-in data, whether or not it's the actual max score
    available in the rubric. For real problems, at least one person should have gotten
    full credit.

    Returns:
        A list of submissions, where each submission is a list of 0/1 assessments.
        e.g. [[0,0,1], [1,1], [0,1,0,0]]
    """
    if len(criterion_assessments) == 0:
        return []

    max_score = max(criterion_assessments, key=lambda x: x.points).points
    by_submission = lambda x: x.submission_uuid
    criterion_assessments.sort(key=by_submission)

    return [[int(a.points == max_score) for a in group]
            for sub_uuid, group in groupby(criterion_assessments, by_submission)]


def points_adjustment(counts):
    """
    Compute an adjustment: e.g. in the example with only 1,3,5 signals used,
    it should be 1->0, 3->1,5->2.

    Args:
        counts: a dict/Counter: original_report -> count

    Returns:
        dict: original_report -> adjusted_report
    """
    max_score = max(counts.keys())
    actual = set(counts.keys())
    def missing_below(k):
        """
        Given a signal, e.g. 3, compute how many signals between 0 and 2 aren't used.
        """
        expected = set(range(k))
        missing_cnt = len(expected - actual)
        return missing_cnt

    adjust = dict((k,k-missing_below(k))
                   for k in counts.keys())
    return adjust


def group_scores_by_submission(criterion_assessments):
    """
    Given a set of assessments for a single criterion, group them by submission and
    convert them to just the point values.

    Removes all unused values. e.g. if the input only has 1, 3, and 5, will map them
    to 0, 1, 2.

    Returns:
        A list of submissions, where each submission is a list of 0/1/...max
         assessments.
        e.g. [[0,3,1], [1,1], [0,1,2,0]]
    """
    if len(criterion_assessments) == 0:
        return []

    by_submission = lambda x: x.submission_uuid

    counts = Counter([a.points for a in criterion_assessments])
    adjust = points_adjustment(counts)

    return [[adjust[a.points] for a in group]
            for sub_uuid, group in groupby(sorted(criterion_assessments, key=by_submission),
                                           by_submission)]



################# Testing code #####################


def check_data(filename):
    """
    WIP!
    """
    all_data = load_assessments(filename)
    just_latest = keep_latest_scores(all_data)


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        # more code, unchanged
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

    check_data(args[0])


if __name__ == "__main__":
    main()
