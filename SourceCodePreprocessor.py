# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from ITokenizer import BaseTokenizer, read_lines_from_model
import re

############ LINNEA ADDED to fix path error 
import os
import sys

CURRENT_PATH = os.path.dirname(__file__)
TOXICR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "./"))
sys.path.insert(1, TOXICR_PATH)
############# LINNEA ADDED to fix path error 


class IdentifierTokenizer(BaseTokenizer):
    def __init__(self):

        self.programming_keywords_list = read_lines_from_model(TOXICR_PATH + '/models/programming_keywords.txt')

    def split_identifiers(self, text):
        result = re.sub('[_]+', ' ', text) # replace underscores with space
        result=re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', result))
        return result

    def remove_keywords(self, text):
        words = text.split()
        resultwords = [word for word in words if word.lower() not in self.programming_keywords_list]
        result = ' '.join(resultwords)
        return result
