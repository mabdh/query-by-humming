# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:16:36 2015

@author: PC
"""
" pip install fuzzywuzzy "

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def getScore(pattern, text):
    score = fuzz.ratio(pattern, text)
    return score