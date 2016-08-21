"""
A quick and dirty cache to save json serializable data and load it back in, converting
strings back to numbers, list keys to tuples, and dicts that look like Counter()s back to
Counter objects.

Author: Victor Shnayder <shnayder@gmail.com>
"""

import json
from collections import Counter
from numbers import Number

def numberize_keys(dct):
    """
    Given a dict with string keys that are secretly numbers,
    return one with them actually numbers (tries int, then float).
    Recursive on nested dicts.
    Also converts dicts with all int values to Counters.
    """
    d = dict()
    for k,v in dct.items():
        if isinstance(v, dict):
            cleaned_v = numberize_keys(v)
        else:
            cleaned_v = v

        try:
            if not isinstance(k, Number):
                d[int(k)] = cleaned_v
            else:
                # it's already a number. Don't mess with it
                d[k] = cleaned_v
        except (ValueError,TypeError):
            try:
                d[float(k)] = cleaned_v
            except:
                d[k] = cleaned_v

    if all(isinstance(x,int) for x in d.values()):
        return Counter(d)

    return d


def uncounterize(dct):
    """
    Given a dict with Counter values, replace them with regular dicts. Works recursively. Returns new dict.
    """
    d = dict()
    for k,v in dct.items():
        if isinstance(v,Counter):
            d[k] = dict(v)
        elif isinstance(v,dict):
            d[k] = uncounterize(v)
        else:
            d[k] = v
    return d


def tuplize_keys(pair):
    """
    Given a pair like [["hi", "there"], {"some": "dict"}],
    return (("hi", "there"), {"some": "dict"})
    """
    k, v = pair
    if isinstance(k, list):
        k = tuple(k)
    return (k,v)

class DirCache(object):
    def __init__(self, cache_dir):
        """
        Args:
            cache_dir: e.g. /foo/bar/baz
        """
        self.cache_dir = cache_dir


    def cache_save(self, filename, data):
        """
        Save data to filename (should end in .json). data should be json-serializable.
        """
        if len(data) == 0:
            raise ValueError("Won't save empty data")
        with open(self.cache_dir + '/' + filename, 'wb') as f:
            json.dump(data, f)


    def cache_load(self, filename):
        """
        Args:
            filename: just the name, no path. Should end in .json.

        Returns:
            Deserialized data, or raises expection if file doesn't exist.
            If the data is a list of key-value pairs, converts back to a dict.
            Parses floats and ints in the json.
        """
        with open(self.cache_dir + '/' + filename, 'r') as f:
            data = json.load(f, parse_float=float,parse_int=int)
            if isinstance(data, list):
                data = dict(map(tuplize_keys, data))
            return numberize_keys(data)
