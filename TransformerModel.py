# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2022
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from random import random
from tqdm import tqdm, trange
import tensorflow as tf
import tensorflow_hub as hub
import  tensorflow_text as text
from official.nlp.optimization import WarmUp

import BertLocator
import random
import numpy as np
import math
from official.nlp import optimization  # to create AdamW optmizer

tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import KFold, train_test_split


def getPTM(model_name):
    if model_name == "BERT":
        return 'bert_en_uncased_L-12_H-768_A-12'
    elif model_name == "ALBERT":
        return 'albert_en_base'
    elif model_name == 'SBERT':
        return 'wiki-books-sst'
    else:
        return None


def df_to_dataset(dataframe, batch_size=16):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


class TransformerModel:
    def __init__(self, X_train=None, Y_train=None,
                 model_name="BERT", load_from_file=None):

        encoder = getPTM(model_name)
        if encoder is None:
            print("Unknown transformer: " + model_name)
            exit(1)
        self.tfhub_handle_encoder = BertLocator.getBERTEncoderURL(encoder)
        self.tfhub_handle_preprocess = BertLocator.getPreprocessURL(encoder)

        print(f'BERT model selected           : {self.tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

        self.bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess)
        self.bert_model = hub.KerasLayer(self.tfhub_handle_encoder)
        self.epochs = 20

        if load_from_file is not None:
            self.steps_per_epoch = 19571  # size of our dataset
            custom_model = self.build_classifier_model()
            config =custom_model.get_config()
            optimizer = self.get_optimizer()
            self._model = tf.keras.models.load_model(load_from_file,  custom_objects={'KerasLayer': hub.KerasLayer,
                                                                                     'AdamWeightDecay': optimizer,
                                                                                      'WarmUp': WarmUp})
        else:
            self.steps_per_epoch = X_train.shape[0]
            self._train(X_train, Y_train)

    def append_numerical_feature(self, row):
        message = row['message']
        if row['profane_count'] > 0:
            message = message + " includes profanity. "
        if row['anger_count'] > 0:
            message = message + " includes anger word. "
        if row['emoticon_count'] > 0:
            message = message + " includes emoticon. "

        return message

    def build_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(2)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        classifier_model = tf.keras.Model(text_input, net)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        optimizer = self.get_optimizer()

        classifier_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)
        # if self.plot is not None:
        #    tf.keras.utils.plot_model(classifier_model, to_file=self.plot, show_shapes=True)
        return classifier_model

    def get_optimizer(self):
        num_train_steps = self.steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1 * num_train_steps)
        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        return optimizer

    def _train(self, X_train, Y_train):
        self._model = self.build_classifier_model()
        if self._model is not None:
            X_train["text"] = X_train.apply(self.append_numerical_feature, axis=1)
            X_train = X_train[["text"]]
            x_new_train, x_validation, y_new_train, y_validation = train_test_split(X_train, Y_train, test_size=0.11115,
                                                                                    random_state=random.randint(1,
                                                                                                                10000))

            x_new_train = x_new_train.reset_index(drop=True)  # resetting index is necessary due to the random split
            x_new_train.fillna(value='', inplace=True)
            y_new_train = y_new_train.reset_index(drop=True)
            x_new_train['target'] = y_new_train.to_numpy()  # merging the two df
            train_ds = df_to_dataset(x_new_train)  # converting to dataset format

            x_validation = x_validation.reset_index(drop=True)
            x_validation.fillna(value='', inplace=True)
            y_validation = y_validation.reset_index(drop=True)
            x_validation['target'] = y_validation.to_numpy()
            validation_ds = df_to_dataset(x_validation)

            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            self._model.fit(x=train_ds, validation_data=validation_ds, epochs=20, callbacks=[es_callback])


    def predict(self, X_values, batch_size=256):

        test_values =X_values["message"].values
        value_count = len(test_values)
        predictions = np.array([])
        num_partitions = math.ceil(value_count / batch_size)
        progress_bar=tqdm(num_partitions)

        for part in trange(0, num_partitions):
            start = part * batch_size
            end = start + batch_size
            if end > value_count:
                end = value_count

            partition = test_values[start:end]
            y_pred = tf.sigmoid(self._model(tf.constant(partition)))
            #y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Threshold: 0.5
            predictions = np.append(predictions, y_pred)

        return predictions

    def save_to_file(self, filename):
        self._model.save(filename)
