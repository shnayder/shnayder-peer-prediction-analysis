#!/usr/bin/env python
"""
Despite what you may suspect, this script is neither arthritis medication nor drug
paraphanelia. It simply removes identifiers from a bunch of joint distributions.

Input:
[[["course_id", "problem_id", "option_id"],
[[0.07857546636517806, 0.035895986433013, 0.041831543244771056],
 [0.035895986433013, 0.054550593555681176, 0.12549462973431316],
 [0.041831543244771056, 0.12549462973431316, 0.4604296212549463]]], ...
 ]

Output, preserves format so parsing code doesn't need to change:
[[[0,0,0],
[[0.07857546636517806, 0.035895986433013, 0.041831543244771056],
 [0.035895986433013, 0.054550593555681176, 0.12549462973431316],
 [0.041831543244771056, 0.12549462973431316, 0.4604296212549463]]], ...
 [[[0,0,1],...],[[0,0,2],...],...
 ]

Author: Victor Shnayder <shnayder@gmail.com>
"""

import sys
import json
from collections import defaultdict

from edx_data.simplify import Mapper

def main(args):
    if len(args) < 2:
        print "usage: clean_joints.py input.json\n\nPrints output to stdout"
        return

    course_mapper = Mapper()
    problem_id_mapper = Mapper()

    # a separate criterion mapper for each (course, problem) pair
    criterion_mappers = defaultdict(Mapper)

    with open(args[1]) as f:
        data = json.load(f)
        new_data = [((course_mapper.get(course_id),
                      problem_id_mapper.get(problem_id),
                      criterion_mappers[(course_id, problem_id)].get(criterion)), joint)
                    for ((course_id, problem_id, criterion), joint)
                    # sort by course_id, problem_id, so all problems for a course and
                    # criteria for a problem are together
                    in sorted(data, key=lambda t: t[0])
                    if 2 <= len(joint)] # remove trivial questions with only one option
        print json.dumps(new_data, indent=4)


if __name__ == "__main__":
    main(sys.argv)
