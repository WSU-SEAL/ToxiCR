# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
import os.path
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from ContractionPreprocessor import expand_contraction, rem_special_sym, remove_url
from ProfanityPreprocessor import PatternTokenizer
from SourceCodePreprocessor import IdentifierTokenizer
from CLEModels import CLEModel
from sklearn.metrics import classification_report
import argparse
import warnings
import random
import timeit

############ LINNEA ADDED to fix path error 
import os
import sys

CURRENT_PATH = os.path.dirname(__file__)
TOXICR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "./"))
sys.path.insert(1, TOXICR_PATH)
############# LINNEA ADDED to fix path error

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_dataframe_from_excel(file):
    dataframe = pd.read_excel(file)
    return dataframe


class ToxiCR:
    def __init__(self, ALGO="RF", embedding="tfidf",
                 model_file=TOXICR_PATH + "/models/code-review-dataset-full.xlsx", split_identifier=False,
                 remove_keywords=False, count_profanity=True,
                 count_anger_words=False,
                 count_emoticon=False,
                 load_pretrained=False):
        self.classifier_model = None
        self.modelFile = model_file
        self.split_identifier = split_identifier
        self.remove_keywords = remove_keywords
        self.count_profanity = count_profanity
        self.count_anger = count_anger_words
        self.count_emoticon = count_emoticon
        self.profanity_checker = PatternTokenizer()
        self.source_code_checker = IdentifierTokenizer()
        self.ALGO = ALGO
        self.embedding = embedding
        self.training_data = read_dataframe_from_excel(model_file)
        self.load_pretrained = load_pretrained

    def preprocess(self, dataframe):
        dataframe["message"] = dataframe.message.astype(str).apply(self.process_text)
        if self.count_profanity:
            dataframe["profane_count"] = dataframe.message.astype(str). \
                apply(self.profanity_checker.count_profanities)
        else:
            dataframe["profane_count"] = 0

        if self.count_anger:
            dataframe["anger_count"] = dataframe.message.astype(str). \
                apply(self.profanity_checker.count_anger_words)
        else:
            dataframe["anger_count"] = 0

        if self.count_emoticon:
            dataframe["emoticon_count"] = dataframe.message.astype(str). \
                apply(self.profanity_checker.emoji_counter)
        else:
            dataframe["emoticon_count"] = 0

    def get_training_data(self):
        self.preprocess(self.training_data)
        return self.training_data

    def __get_pretrained_model(self):
        return True

    def process_text(self, text):
        # mandatory preprocessing
        processed_text = remove_url(text)
        processed_text = expand_contraction(processed_text)
        processed_text = self.profanity_checker.process_text(processed_text)
        processed_text = rem_special_sym(processed_text)
        # optional preprocessing
        if self.split_identifier:
            processed_text = self.source_code_checker.split_identifiers(processed_text)

        if self.remove_keywords:
            processed_text = self.source_code_checker.remove_keywords(processed_text)
        #print( processed_text)
        return processed_text

    def init_predictor(self):
        if self.load_pretrained:
            filename = self.getPTMName()
            
            loadstatus = self.load_pretrained_model(filename)
            if loadstatus:
                print("Successfully loaded pretrained model from "+filename)
                return
            else:
                print("Unable to load pretrained model: "+filename)
        self.__train_predictor()

    def getPTMName(self):
        ALGO=self.ALGO
        filename = TOXICR_PATH + "/pre-trained/model-" + ALGO + "-" + str(self.embedding) + "-profane-" \
 + str(self.count_profanity) + "-keyword-" + str(self.remove_keywords) + "-split-" \
                   + str(self.split_identifier)
        if ((ALGO == "CNN") | (ALGO == "LSTM") | (ALGO == "GRU") | (ALGO == "biLSTM")):
            filename = filename + ".h5"
        elif(ALGO =="BERT"):
            filename = filename + ".h5"
        elif ((ALGO == "RF") | (ALGO == "GBT") | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR")):
            filename = filename + ".pickle"
               
        print("getPTMName ", filename)
            
        return filename

    def __train_predictor(self):
        self.preprocess(self.training_data)
        X_train = self.training_data[["message", "profane_count", "anger_count", "emoticon_count"]]
        Y_train = self.training_data[['is_toxic']]
        # train model using full dataset
        self.get_model(X_train, Y_train)

    def train_for_tuning(self):
        self.preprocess(self.training_data)
        X_train = self.training_data[["message", "profane_count", "anger_count", "emoticon_count"]]
        Y_train = self.training_data[['is_toxic']]
        # train model using full dataset
        self.get_model(X_train, Y_train, tuning=True)

    def save_trained_model(self):
        ALGO = self.ALGO
        filename = self.getPTMName()
        if ((ALGO == "BERT") | (ALGO == "ALBERT") | (ALGO == "SBERT") | (ALGO == "CNN") | (ALGO == "LSTM") | \
                (ALGO == "GRU") | (ALGO == "biLSTM")):
            self.classifier_model.save_to_file(filename)
        elif ((ALGO == "RF") | (ALGO == "GBT") | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR")):
            pickle.dump(self.classifier_model, open(filename, "wb"))
        print("Model stored as: "+filename)

    def load_pretrained_model(self, filename):
        #if not os.path.exists(filename):
        #    print("File: "+ filename +" not exists!")
         #   return False
        if filename.endswith(".pickle"):
            self.classifier_model = pickle.load(open(filename, "rb"))
            return True
        ALGO = self.ALGO
        try:
            if ((ALGO == "CNN") | (ALGO == "LSTM") |
                    (ALGO == "GRU") | (ALGO == "biLSTM")):
                import DNNModels
                self.classifier_model = DNNModels.DNNModel(algo=ALGO, embedding=self.embedding,
                                                           load_from_file=filename)
                return True
            elif (ALGO == "BERT") | (ALGO == "ALBERT") | (ALGO == "SBERT"):
                from TransformerModel import TransformerModel
                self.classifier_model = TransformerModel(load_from_file=filename)
                return True
        except Exception as e:
            print(e)
            return False

    def get_model(self, X_train, Y_train, tuning=False):
        ALGO = self.ALGO
        if (ALGO == "RF") | (ALGO == "GBT") | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR"):
            self.classifier_model = CLEModel(X_train=X_train, Y_train=Y_train, algo=self.ALGO, tuning=tuning)
        elif (ALGO == "BERT") | (ALGO == "ALBERT") | (ALGO == "SBERT"):
            from TransformerModel import TransformerModel
            self.classifier_model = TransformerModel(X_train=X_train, Y_train=Y_train)
        elif (ALGO == "CNN") | (ALGO == "LSTM") | (ALGO == "GRU") | (ALGO == "biLSTM"):
            import DNNModels

            self.classifier_model = DNNModels.DNNModel(X_train=X_train,
                                                       Y_train=Y_train,
                                                       algo=ALGO, embedding=self.embedding)
        else:
            print("Unknown algorithm: "+ALGO)
            exit(1)

        return self.classifier_model

    def get_toxicity_probability(self, texts):
        dataframe = pd.DataFrame(texts, columns=['message'])
        self.preprocess(dataframe)
        #print(dataframe)
        results = self.classifier_model.predict(dataframe)
        return results


def get_misclassifications(dataframe, labels, predictions):
    predictions = pd.DataFrame(data=predictions, columns=["predicted"])
    newdf = dataframe.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    merged_df = pd.concat([newdf, predictions], axis=1)
    merged_df = pd.concat([merged_df, labels_reset], axis=1)

    misclassified_df = merged_df[(merged_df["predicted"] != merged_df["is_toxic"])]
    return misclassified_df


def ten_fold_cross_validation(toxicClassifier, rand_state):
    dataset = toxicClassifier.get_training_data()

    dataset.to_excel("count-profane.xlsx")

    skf =StratifiedKFold(n_splits=10, shuffle=True, random_state=rand_state)

    kf = KFold(n_splits=10, shuffle=True, random_state=rand_state)
    results = ""

    count = 1
    all_misclassifications = pd.DataFrame()

    for train_index, test_index in skf.split(dataset, dataset["is_toxic"]):
        start = timeit.default_timer()
        print("Using split-" + str(count) + " as test data..")
        results = results + str(count) + "," + ALGO + ","

        X_train, X_test = dataset.loc[train_index, ["message", "profane_count", "anger_count", "emoticon_count"]], \
                          dataset.loc[test_index, ["message", "profane_count", "anger_count", "emoticon_count"]]
        Y_train, Y_test = dataset.loc[train_index, "is_toxic"], dataset.loc[test_index, "is_toxic"]
        classifier_model = toxicClassifier.get_model(X_train, Y_train)

        Y_prob = classifier_model.predict(X_test)
        predictions = [1 if pred >= 0.5 else 0 for pred in Y_prob]
        misclassified = get_misclassifications(X_test, Y_test, predictions)

        stop = timeit.default_timer()
        time_elapsed = stop - start

        if len(all_misclassifications.columns) == 0:
            all_misclassifications = misclassified
            print("Misclassification count: " + str(len(all_misclassifications)))
        else:
            all_misclassifications = pd.concat([all_misclassifications, misclassified], axis=0)
            print("Misclassification count: " + str(len(misclassified)))

        precision_1 = precision_score(Y_test, predictions, pos_label=1)
        recall_1 = recall_score(Y_test, predictions, pos_label=1)
        f1score_1 = f1_score(Y_test, predictions, pos_label=1)

        precision_0 = precision_score(Y_test, predictions, pos_label=0)
        recall_0 = recall_score(Y_test, predictions, pos_label=0)
        f1score_0 = f1_score(Y_test, predictions, pos_label=0)
        accuracy = accuracy_score(Y_test, predictions)

        results = results + str(precision_0) + "," + str(recall_0) + "," + str(f1score_0)
        results = results + "," + str(precision_1) + "," + str(recall_1) + "," + str(f1score_1) + \
                  "," + str(accuracy) + "," + str(time_elapsed) + "\n"

        print(classification_report(Y_test, predictions))
        count += 1

    return (results, all_misclassifications)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ToxiCR: A supervised Toxicity Analysis tool for the SE domain')

    parser.add_argument('--algo', type=str,
                        help='Classification algorithm. Choices are: RF| DT| SVM| LR| GBT| CNN|' +
                             ' LSTM| GRU| biLSTM|  BERT| ALBERT| SBERT',
                        default="RF")

    parser.add_argument('--repeat', type=int, help='Iteration count', default=2)
    parser.add_argument('--embed', type=str,
                        help='Word embedding Choices are: tfidf| fasttext | word2vec | glove | bert',
                        default="tfidf")

    parser.add_argument('--split', help='Split identifiers', action='store_true', default=False)
    parser.add_argument('--keyword', help='Remove programming keywords', action='store_true', default=False)
    parser.add_argument('--profanity', help='Count profane words', action='store_true', default=False)
    parser.add_argument('--anger', help='Count anger words', action='store_true', default=False)
    parser.add_argument('--emoticon', help='Count emoticons', action='store_true', default=False)
    parser.add_argument('--retro', help='Print missclassifications',
                        action='store_true', default=False)  # default False, will not write
    parser.add_argument('--mode', type=str,
                        help='Execution mode. Choices are: eval | pretrain | tuning',
                        default="eval")

    args = parser.parse_args()

    print(args)
    ALGO = str(args.algo).upper()
    REPEAT = args.repeat
    embedding = args.embed
    mode = args.mode
    toxicClassifier = ToxiCR(split_identifier=args.split, remove_keywords=args.keyword, count_profanity=args.profanity,
                             ALGO=ALGO, count_emoticon=args.emoticon,
                             count_anger_words=args.anger,
                             embedding=embedding)

    if mode == 'tuning':
        if (ALGO == 'RF') | (ALGO == 'DT'):
            toxicClassifier.train_for_tuning()
            exit(0)
        else:
            print("Hyperparameter search is not implemented for the selected algorithm!")
            exit(0)
    elif mode == 'pretrain':
        toxicClassifier.init_predictor()
        toxicClassifier.save_trained_model()
        exit(0)

    timers = []

    filename = "cross-validation-" + ALGO + "-" + str(args.embed) + "-profane-" \
               + str(args.profanity) + "-keyword-" + str(args.keyword) + "-split-" + str(args.split) + ".csv"
    training_log = open(filename, 'w')
    training_log.write("Fold,Algo,precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,accuracy,time\n")

    random.seed(999)
    for k in range(0, REPEAT):
        print(".............................")
        print("Run# {}".format(k))
        (results, misclassified) = ten_fold_cross_validation(toxicClassifier, random.randint(1, 10000))
        training_log.write(results)
        training_log.flush()
        if (args.retro & (k == 0)):
            misclassified.to_excel(ALGO + "-" + str(args.embed) + "-profane-" \
                                   + str(args.profanity) + "-keyword-" + str(args.keyword) + "-split-" + str(args.split)
                                   + "_misclassified.xlsx")

    ##########################

    training_log.close()
