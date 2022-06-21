# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import pickle
import warnings

import numpy as np
import tensorflow as tf
# print(tf.__version__)
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.keras import backend as K
import  random

# from keras.models import Model


warnings.filterwarnings('ignore')

EMBEDDING_word2vec = './embeddings/word2vec-googlenews-300d.txt'
EMBEDDING_FastText = './embeddings/crawl-300d-2M.vec'
EMBEDDING_GloVe = './embeddings/glove.840B.300d.txt'

EMBEDDING_word2vec_loaded = None
EMBEDDING_FastText_loaded = None
EMBEDDING_GloVe_loaded = None

count_based_features = ["profane_count", "anger_count", "emoticon_count"]


def load_embedding_index(embedding_file):
    print("Loading embedding file: " + embedding_file)
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file, encoding='UTF8'))
    print("Embedding file loaded")
    return embeddings_index


def get_embedding_index(embedding_type):
    global EMBEDDING_FastText_loaded, EMBEDDING_GloVe_loaded, EMBEDDING_word2vec_loaded
    if embedding_type == "fasttext":
        if None == EMBEDDING_FastText_loaded:
            EMBEDDING_FastText_loaded = load_embedding_index(EMBEDDING_FastText)
        return EMBEDDING_FastText_loaded
    elif embedding_type == "glove":
        if None == EMBEDDING_GloVe_loaded:
            EMBEDDING_GloVe_loaded = load_embedding_index(EMBEDDING_GloVe)
        return EMBEDDING_GloVe_loaded
    elif embedding_type == "word2vec":
        if None == EMBEDDING_word2vec_loaded:
            EMBEDDING_word2vec_loaded = load_embedding_index(EMBEDDING_word2vec)
        return EMBEDDING_word2vec_loaded
    else:
        print("Invalid or unsupported embedding: "+embedding_type)
        exit(1)


def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind]


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


class DNNModel:
    def __init__(self, X_train=None, Y_train=None, algo="CNN", embedding="fasttext",
                 max_features=5000, maxlen=500,
                 embedding_size=300, load_from_file=None):
        self.max_features = max_features
        num_count_features = len(count_based_features)
        self.maxlen = maxlen + num_count_features
        self.embed_size = embedding_size
        self.embedding_type = embedding

        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
        K.set_session(tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf))

        self.algo = algo
        if load_from_file is not None:
            self.tokenizer = pickle.load(open(load_from_file+".pickle", "rb"))
            self.model=tf.keras.models.load_model(load_from_file)
        else:
            self.tokenizer = text.Tokenizer(num_words=max_features)
            self.tokenizer.fit_on_texts(X_train['message'])
            self.embedding_matrix = self._create_embedding()
            self.model = self._train(X_train, Y_train)

    ##this is DPCNN model
    def _create_model_DPCNN(self):
        from tensorflow.keras.models import Model
        filter_nr = 64
        filter_size = 3
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 256
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)

        comment = tf.keras.layers.Input(shape=(self.maxlen,))
        emb_comment = tf.keras.layers.Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix],
                                                trainable=train_embed)(comment)
        emb_comment = tf.keras.layers.SpatialDropout1D(spatial_dropout)(emb_comment)

        block1 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.PReLU()(block1)
        block1 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = tf.keras.layers.Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(
            emb_comment)
        resize_emb = tf.keras.layers.PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        block1_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = tf.keras.layers.BatchNormalization()(block2)
        block2 = tf.keras.layers.PReLU()(block2)
        block2 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = tf.keras.layers.BatchNormalization()(block2)
        block2 = tf.keras.layers.PReLU()(block2)

        block2_output = add([block2, block1_output])
        block2_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = tf.keras.layers.BatchNormalization()(block3)
        block3 = tf.keras.layers.PReLU()(block3)
        block3 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = tf.keras.layers.BatchNormalization()(block3)
        block3 = tf.keras.layers.PReLU()(block3)

        block3_output = add([block3, block2_output])
        block3_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = tf.keras.layers.BatchNormalization()(block4)
        block4 = tf.keras.layers.PReLU()(block4)
        block4 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = tf.keras.layers.BatchNormalization()(block4)
        block4 = tf.keras.layers.PReLU()(block4)

        block4_output = add([block4, block3_output])
        block4_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = tf.keras.layers.BatchNormalization()(block5)
        block5 = tf.keras.layers.PReLU()(block5)
        block5 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = tf.keras.layers.BatchNormalization()(block5)
        block5 = tf.keras.layers.PReLU()(block5)

        block5_output = add([block5, block4_output])
        block5_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

        block6 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
        block6 = tf.keras.layers.BatchNormalization()(block6)
        block6 = tf.keras.layers.PReLU()(block6)
        block6 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
        block6 = tf.keras.layers.BatchNormalization()(block6)
        block6 = tf.keras.layers.PReLU()(block6)

        block6_output = add([block6, block5_output])
        block6_output = tf.keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

        block7 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
        block7 = tf.keras.layers.BatchNormalization()(block7)
        block7 = tf.keras.layers.PReLU()(block7)
        block7 = tf.keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
        block7 = tf.keras.layers.BatchNormalization()(block7)
        block7 = tf.keras.layers.PReLU()(block7)

        block7_output = add([block7, block6_output])
        output = tf.keras.layers.GlobalMaxPooling1D()(block7_output)

        output = tf.keras.layers.Dense(dense_nr, activation='linear')(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.PReLU()(output)
        output = tf.keras.layers.Dropout(dense_dropout)(output)

        ### one layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

        model = Model(comment, output)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])

        return model

    ## this is bidirectional GRU  model
    def _create_GRU_Model(self):
        inp = tf.keras.layers.Input(shape=(self.maxlen,))
        x = tf.keras.layers.Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix])(inp)
        x = tf.keras.layers.SpatialDropout1D(0.2)(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(80, return_sequences=True, recurrent_dropout=0.1))(x)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        conc = tf.keras.layers.concatenate([avg_pool, max_pool])

        x = tf.keras.layers.Dense(1, activation="sigmoid")(conc)

        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    ## this is bi directional LSTM model
    def _create_biLSTM_Model(self):
        inp = tf.keras.layers.Input(shape=(self.maxlen,))
        x = tf.keras.layers.Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix])(inp)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        x = tf.keras.layers.Dense(50, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _create_ATTN_Model(self):
        input = tf.keras.layers.Input(shape=(self.maxlen + 1,))
        x = tf.keras.layers.Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix])(input)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

        rnn_outs = tf.keras.layers.Dense(50, activation="relu")(x)
        rnn_outs = tf.keras.layers.Dropout(0.2)(rnn_outs)

        attention_vector = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(rnn_outs)
        attention_vector = tf.keras.layers.Reshape((self.maxlen + 1,))(attention_vector)
        attention_vector = tf.keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
        attention_output = tf.keras.layers.Dot(axes=1)([rnn_outs, attention_vector])

        fc = tf.keras.layers.Dense(64, activation='relu')(attention_output)
        fc = tf.keras.layers.Dropout(0.5)(fc)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(fc)

        model = Model(inputs=input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    ## this is unidirectional LSTM model
    def _create_uniLSTM_Model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(self.max_features, self.embed_size, input_length=self.maxlen,
                                            weights=[self.embedding_matrix], trainable=False))
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.LSTM(self.embed_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, \
                                       kernel_regularizer=regularizers.l2(0.005), \
                                       bias_regularizer=regularizers.l2(0.005)))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(512, activation='relu', \
                                        kernel_regularizer=regularizers.l2(0.001), \
                                        bias_regularizer=regularizers.l2(0.001), ))
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Dense(8, activation='relu', \
                                        kernel_regularizer=regularizers.l2(0.001), \
                                        bias_regularizer=regularizers.l2(0.001), ))
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _prepare_df(self, df):
        df_msg = df['message']

        num_count_features = len(count_based_features)

        df_manual_features = df[[c for c in df.columns if c in count_based_features]]
        manual_feature_list = df_manual_features.to_numpy()

        df_msg = self.tokenizer.texts_to_sequences(df_msg)
        df_msg = sequence.pad_sequences(df_msg, maxlen=self.maxlen - num_count_features)
        df_merged = np.hstack((df_msg, manual_feature_list))
        return df_merged

    def _create_embedding(self):
        embeddings_index = get_embedding_index(self.embedding_type)

        word_index = self.tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words, self.embed_size))
        for word, i in word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def _train(self, X_train, Y_train):

        model = None
        if self.algo == "CNN":
            model = self._create_model_DPCNN()
        elif self.algo == "GRU":
            model = self._create_GRU_Model()
        elif self.algo == "LSTM":
            model = self._create_uniLSTM_Model()
        elif self.algo == "biLSTM":
            model = self._create_biLSTM_Model()
        elif self.algo == "ATTN":
            self.attention=True
            model = self._create_ATTN_Model()

        X_train_vector = self._prepare_df(X_train)

        if model is not None:
            ##convert Y_train to np.ndaaray
            Y_train = Y_train.values

            x_new_train, x_val, y_new_train, y_val = train_test_split(X_train_vector, Y_train, test_size=0.11115,
                                                                      random_state=random.randint(1,10000))

            es_callback = EarlyStopping(monitor='val_loss', patience=3)
            model.fit(x_new_train, y_new_train, batch_size=128, epochs=40, validation_data=(x_val, y_val),
                      callbacks=[es_callback])

        return model

    def predict(self, X_test):
        X_test_mapped = self._prepare_df(X_test)
        y_pred = self.model.predict(X_test_mapped, batch_size=32)
        return y_pred

    def attention(self, X_test):
        X_test_mapped = self._prepare_df(X_test)
        y_pred = self.model.predict(X_test_mapped, batch_size=32)

        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return y_pred

    def save_to_file(self, filename):
        self.model.save(filename)
        #Need to store tokenizer as well for processing text
        pickle.dump(self.tokenizer, open(filename+".pickle", "wb"))
