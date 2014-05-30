#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pure_lda.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2014-05-07
'''
import string 
import sys


stop = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 'her',
 'hers',
 'herself',
 'it',
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 'should',
 'now']

delEStr = string.punctuation+''+string.digits
delEStr = delEStr.replace('\'', '')
table = string.maketrans(delEStr, ' '*len(delEStr))
def load_data():
    for line in sys.stdin:
        if len(line.strip().split()) > 3:
            total = line.strip().split(' ',3)
            user =total [0]
            item = total[1]
            rate = total[2]
            line = total[-1]
            line = line.translate(table)
            line = line.lower()
            line = line.replace('  ', ' ')
            print user, item, rate,' '.join([i for i in line.split() if i not in stop])

if __name__ == "__main__":
    load_data()
