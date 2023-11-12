# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from ToxiCR import ToxiCR

#Best configurations for each algorithms are listed.
# Pretrained models for the best configurations are also included.
#Random Forest, recommended, if you do not have GPU
#toxicClassifier=ToxiCR(ALGO="RF", count_profanity=True, remove_keywords=False,split_identifier=False,
#                   embedding="tfidf", load_pretrained=True)

#BiLSTM
# toxicClassifier=ToxiCR(ALGO="biLSTM", count_profanity=True, remove_keywords=False,split_identifier=True,
#                                       embedding="fasttext", load_pretrained=True)

#LSTM
#toxicClassifier=ToxiCR(ALGO="LSTM", count_profanity=True, remove_keywords=True,split_identifier=True,
#                                   embedding="glove", load_pretrained=True)

# Gated Recurrent Unit (GRU)
#toxicClassifier=ToxiCR(ALGO="GRU", count_profanity=True, remove_keywords=False,split_identifier=True,
#                                 embedding="glove", load_pretrained=True)

# Deep Pyramid Convolutional neural networks (DPCNN)
# toxicClassifier=ToxiCR(ALGO="CNN", count_profanity=True, remove_keywords=False,split_identifier=False,
#                        embedding="fasttext", load_pretrained=True)

#Decision Tree (DT)
# toxicClassifier=ToxiCR(ALGO="DT", count_profanity=True, remove_keywords=True,split_identifier=False,
#                          embedding="tfidf", load_pretrained=True)

#Logistic Regression (LR)
# toxicClassifier=ToxiCR(ALGO="LR", count_profanity=True, remove_keywords=True, split_identifier=False,
#                        embedding="tfidf", load_pretrained=True)

#Gradient Boosting Tree (GBT)
# toxicClassifier=ToxiCR(ALGO="GBT", count_profanity=True, remove_keywords=True, split_identifier=False,
#                              embedding="tfidf", load_pretrained=True)

# Support Vector Machine(SVM)
# toxicClassifier=ToxiCR(ALGO="SVM", count_profanity=True, remove_keywords=False,split_identifier=True,
#                        embedding="fasttext", load_pretrained=True)

# Bert, the best performing model
toxicClassifier = ToxiCR(
    ALGO="BERT",
    count_profanity=False,
    remove_keywords=True,
    split_identifier=False,
    embedding="tfidf",
    load_pretrained=True
)

toxicClassifier.init_predictor()

sentences = [
    "go fuck yourself",
    "this is crap",
    "thank you for the information",
    "yeah that sucked, fixed, Done.",
    "Crap, this is an artifact of a previous revision. It's simply the last time a change was made to Tuskar's cloud.",
    "Ah damn I misread the bug -_-",
    "wtf...",
    "I appreciate your help.",
    "fuuuuck",
    "what the f*ck",
    "absolute shit",
    "Get the hell outta here",
     "shi*tty code",
    "you are an absolute b!tch",
    "Nothing particular to worry about",
    "You need to kill the process for it to work"
]


results = toxicClassifier.get_toxicity_probability(sentences)
for i in range(len(sentences)):
    print("\"" + sentences[i] + "\" ->" + str(results[i]))  # probablity of being toxic.
