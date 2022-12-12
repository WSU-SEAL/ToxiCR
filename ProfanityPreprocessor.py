# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import copy
import re
from nltk import word_tokenize
from ITokenizer import BaseTokenizer, read_lines_from_model

############ LINNEA ADDED to fix path error 
import os
import sys

CURRENT_PATH = os.path.dirname(__file__)
TOXICR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "./"))
sys.path.insert(1, TOXICR_PATH)
############# LINNEA ADDED to fix path error 

emoticon_list = {':)', ':(', ':/', ':O', ':o', ':-(', '>:)', '<|:O', ':?:', ':-|', '|-O',
                 '</3', ':(', ':-)', ':-*', ':D', '<3', ':S', ':P', ';)', ';-)', ':-o'}

RE_PATTERNS = {

    ' fuck ':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*',
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'

        ],

    ' crap ':
        [
            '(c)(r|[^a-z0-9 ])(a|[^a-z0-9 ])(p|[^a-z0-9 ])([^ ])*',
            '(c)([^a-z]*)(r)([^a-z]*)(a)([^a-z]*)(p)',
            ' c[!@#\$%\^\&\*]*r[!@#\$%\^&\*]*p', 'cr@p', 'c r a p',

        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
        ],

    ' ass hole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'
        ],

    ' bitch ':
        [
            'bitches', 'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h'
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' transgender':
        [
            'transgender'
        ],

    ' gay ':
        [
            'gay', 'homo'
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'd i c k'
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' bull shit ':
        [
            'bullsh\*t', 'bull\$hit', 'bull sh.t'
        ],

    ' jerk ':
        [
            'jerk'
        ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots' 'i d i o t'
        ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t', 'sh\*tty',
            'sh\*ty', 'sh\*t'
        ],

    ' shit hole ':
        [
            'shythole', 'sh.thole'
        ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
        ],

    ' rape ':
        [
            'raped'
        ],

    ' dumb ass':
        [
            'dumbass', 'dubass'
        ],

    ' ass head':
        [
            'butthead'
        ],

    ' sex ':
        [
            'sexy', 's3x', 'sexuality'
        ],

    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
        ],

    ' shut the fuck up':
        [
            ' stfu' '^stfu'
        ],

    ' for your fucking information':
        [
            ' fyfi', '^fyfi'
        ],
    ' get the fuck off':
        [
            'gtfo', '^gtfo'
        ],

    ' oh my fucking god ':
        [
            ' omfg', '^omfg'
        ],

    ' what the hell ':
        [
            ' wth', '^wth'
        ],

    ' what the fuck ':
        [
            ' wtf', '^wtf'
        ],
    ' son of bitch ':
        [
            ' sob ', '^sob '
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses', '(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)',
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' mother fucker':
        [
            ' motha f', ' mother f', 'motherucker', ' mofo', ' mf ',
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],

    ' haha ':
        [
            'ha\*\*\*ha',
        ],
    # ' what the fuck ':
    # [
    #     ' wtf',
    # ],
}



class PatternTokenizer(BaseTokenizer):
    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\*\+\?\&\_\-,\.' ]", patterns=RE_PATTERNS,
                 remove_repetitions=True):
        self.lower = lower
        self.patterns = patterns
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions
        self.profanity_list = read_lines_from_model(TOXICR_PATH + '/models/profane-words.txt')
        self.anger_word_list = read_lines_from_model(TOXICR_PATH + '/models/anger-words.txt')

    def process_text(self, text):
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x

    def count_profanities(self, text):
        count = 0
        words = word_tokenize(text)
        for profane_word in self.profanity_list:
            if profane_word in words:
                # print(profane_word)
                count = count + 1
        return count

    def count_anger_words(self, text):
        count = 0
        words = word_tokenize(text)
        for anger_word in self.anger_word_list:
            if anger_word in words:
                count = count + 1
        return count

    def emoji_counter(self, text):
        return len(list(filter(lambda x: x in emoticon_list, text.split(' '))))

    def process_ds(self, ds):
        ### ds = Data series

        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()

        # remove special chars
        if self.initial_filters is not None:
            ds = ds.str.replace(self.initial_filters, ' ')

        # looooooooooser = loser
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            ds = ds.str.replace(pattern, r"\1")

        for target, patterns in self.patterns.items():
            for pat in patterns:
                ds = ds.str.replace(pat, target)

        ds = ds.str.replace(r"[^a-z' ]", ' ')

        return ds.str.split()

    def _preprocess(self, text):
        # lower
        if self.lower:
            text = text.lower()

        # remove special chars
        if self.initial_filters is not None:
            text = re.sub(self.initial_filters, ' ', text)

        # neeeeeeeeeerd => nerd
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            text = pattern.sub(r"\1", text)
        return text
