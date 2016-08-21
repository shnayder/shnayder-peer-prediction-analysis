#!/usr/bin/env python

import csv
import getopt
import gzip
import sys
import json

from datetime import datetime
from operator import itemgetter
from itertools import groupby, imap
from collections import namedtuple, Counter

from edx_data import HEADERS, SIMPLIFY, Assessment

class Mapper(object):
    """
    Map arbitrary (hashable) keys to to sequential ints. e.g.:

    mapper = Mapper()
    >>> mapper.get("something")
    0
    >>> mapper.get("abc")
    1
    >>> mapper.get("cdf")
    2
    >>> mapper.get("abc")
    1
    >>> mapper.get("xyz")
    3
    >>> mapper.map()
    {1:"abc", 2:"cdf", 3:"xyz"}
    """
    def __init__(self):
        self.next_num = 0
        self.revmap = {}  # old id -> new shortened id
        self.fwdmap = {}  # shortened id -> old id

    def get(self, old_id):
        if old_id in self.revmap:
            return self.revmap[old_id]
        # otherwise, make a new key
        self.fwdmap[self.next_num] = old_id
        self.revmap[old_id] = self.next_num
        self.next_num += 1
        return self.next_num - 1

    def map(self):
        return self.fwdmap

def simplify(infile, outfile, mapfile):
    """
    Read a csv of assessments from filename, unzipping if it ends in '.gz',
    and simplify it. Replaces values in SIMPLIFY with sequential integers,
    saving a map in mapfile as a json dict:

    {"item_id": {1: "old-long-item_id", 2: ...},
     "student_id": {1: ...., 2: ...},
     ...}
    """
    mappers = {}
    for k in SIMPLIFY:
        mappers[k] = Mapper()

    if infile.endswith('.gz'):
        fin = gzip.GzipFile(infile, 'rb')
    else:
        fin = open(infile)

    if outfile.endswith('.gz'):
        fout = gzip.GzipFile(outfile, 'wb')
    else:
        fout = open(outfile, 'wb')


    reader = csv.reader(fin, dialect='MyDialect')
    writer = csv.writer(fout, dialect='MyDialect')

    # Skip the header row
    reader.next()
    writer.writerow(HEADERS)

    skips = []
    for row in reader:
        try:
            elt = Assessment(*row)
            elt = elt._replace(points=int(elt.points),
                               scored_at=datetime.strptime(elt.scored_at,
                                                           '%Y-%m-%d %H:%M:%S'))

            replacements = dict((k, mappers[k].get(getattr(elt,k)))
                                for k in SIMPLIFY)
            elt = elt._replace(**replacements)

            writer.writerow(elt)

        except ValueError:
            # Skip rows that don't have an int for points, or improper time
            skips.append(reader.line_num)
            continue
    if len(skips) > 0:
        print "Skipped {} input rows".format(len(skips))

    fin.close()
    fout.close()

    maps = dict((k, mappers[k].map()) for k in SIMPLIFY)
    with open(mapfile, 'wb') as fmap:
        json.dump(maps, fmap)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    infile, outfile, mapfile = argv[1:4]

    simplify(infile, outfile, mapfile)

if __name__ == "__main__":
    main()
