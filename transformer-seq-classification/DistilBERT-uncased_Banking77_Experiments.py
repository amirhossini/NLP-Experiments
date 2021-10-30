"""
Project:__ Intent Recognition
Sub-prj:__ Parametric Evaluation - _number of labels & data fraction_
Experm:__ Distilbert-base-uncased
Status:__ Version 2.6
    - Developed Callnack function for calculation of f1_score, precision and recall
    - Reconfigured for instance level sampling
    - Compilation is set out of loop
    - Full experiment is run
Contact:__ Amir Hossini
Dev Dat:__ Oct 29, 2021
"""

## Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import transformers
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Tensorflow version: {tf.__version__}")
print(f"Sklearn version: {sklearn.__version__}")
print(f"Transformers version: {transformers.__version__}")

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## I/O files and params
fl_train        = '../datasets/banking77/train.csv'
fl_test         = '../datasets/banking77/test.csv'

checkpoint      ='distilbert-base-uncased'

experiment_grid = {
    'n_labels'  : [3,4,8,16,32,64],
    'n_train_inst': [3,4,8,16,32,64]
}

seed = 99
BIG_int = 10**21

## Functions

class Metrics(Callback):
    def __init__(self, validation_data):
        super(Metrics, self).__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(tf.nn.softmax(self.model.predict(self.validation_data[0]).logits), 1)
        val_targ = self.validation_data[1]
        val_f1 = f1_score(val_targ, val_predict, average='macro')
        val_recall = recall_score(val_targ, val_predict, average='macro')
        val_precision = precision_score(val_targ, val_predict, average='macro')
        logs['val_f1'] = val_f1
        logs['val_recall'] = val_recall
        logs['val_precision'] = val_precision
        print(
            f'— val_f1: {round(val_f1, 4)} — val_precision: {round(val_precision, 4)} — val_recall: {round(val_recall, 4)}')
        return


def select_data_instances(train, test, col_label, n_labels=None, n_train_inst=None):
    if n_labels is None:
        n_labels = len(train.loc[:, col_label].unique())
    if n_train_inst is None:
        n_train_inst = BIG_int

    select_labels = np.array(train.loc[:, col_label].value_counts()[0:n_labels].index)
    select_train = train.loc[train[col_label].isin(select_labels), :].reset_index(drop=True)
    subset_train = pd.DataFrame(columns=select_train.columns)
    for label in select_labels:
        temp_train = select_train.loc[select_train[col_label] == label, :].reset_index(drop=True)
        select_indx = np.random.choice(range(len(temp_train)), min(n_train_inst, len(temp_train)))
        temp_train = temp_train.iloc[select_indx, :]
        subset_train = subset_train.append(temp_train)
    subset_train = subset_train.sample(frac=1).reset_index(drop=True)
    select_test = test.loc[test[col_label].isin(select_labels), :].reset_index(drop=True)
    select_test = select_test.sample(frac=1).reset_index(drop=True)
    return subset_train, select_test


def factorize_data(train, test, col_label, return_keys=False):
    train_labels, levels = pd.factorize(train.loc[:, col_label])
    categs = pd.concat([train.loc[:, col_label], pd.Series(train_labels)], axis=1).groupby(col_label).mean().index
    labels = pd.concat([train.loc[:, col_label], pd.Series(train_labels)], axis=1).groupby(col_label).mean().values.T[0]
    label_keys = dict(zip(categs, labels))
    test_labels = test.loc[:, col_label].map(lambda x: label_keys[x])
    train.loc[:, col_label] = train_labels
    test.loc[:, col_label] = test_labels
    if return_keys:
        return train, test, label_keys
    else:
        return train, test


def tokenize_encode_data(train, test, col_text, col_label, checkpoint,
                         truncation_flag=True, padding_flag=True, return_tensors_opt='np'):
    train_sentences = list(train.loc[:, col_text])
    test_sentences = list(test.loc[:, col_text])
    train_labels = list(train.loc[:, col_label])
    test_labels = list(test.loc[:, col_label])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_encodings = tokenizer(train_sentences, truncation=truncation_flag,
                                padding=padding_flag, return_tensors=return_tensors_opt)
    test_encodings = tokenizer(test_sentences, truncation=truncation_flag,
                               padding=padding_flag, return_tensors=return_tensors_opt)
    return train_encodings.data, train_labels, test_encodings.data, test_labels


def transformer_model_compile(checkpoint, n_labels, lr=5e-5):
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=n_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def transformer_model_fit(model, train_encodings, train_labels, test_encodings, test_labels, epochs=10, batch_size=64):
    history = model.fit(train_encodings, np.array(train_labels), epochs=epochs, batch_size=batch_size,
                        validation_data=(test_encodings, np.array(test_labels)), callbacks=[Metrics(validation_data)])
    return history, model


def make_plots(df):
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (14, 9)})
    sns.set_context("talk")
    sns.lineplot(x="sample_size", y="val_f1", hue="num_categ", marker='o', data=df, legend="full")
    plt.show()
    sns.lineplot(hue="sample_size", y="val_f1", x="num_categ", marker='o', data=df, legend="full")

## Modeling Loop
np.random.seed(seed)
train_orig = pd.read_csv(fl_train)
test_orig = pd.read_csv(fl_test)

Big_Matrix = pd.DataFrame(columns=['num_categ', 'sample_size', 'val_f1'])

iexp = 0
for ilab in experiment_grid['n_labels']:
    compiled_model = transformer_model_compile(checkpoint, ilab)
    for i_ninst in experiment_grid['n_train_inst']:
        iexp += 1
        train, test = select_data_instances(train_orig, test_orig, 'category', ilab, i_ninst)
        train, test = factorize_data(train, test, 'category')
        print(f'\nExperiment {iexp} with {ilab} labels and {i_ninst} instances')
        print(f'\nsize of training dataset: {len(train)}')
        print(f'size of testing dataset: {len(test)}')

        train_encodings, train_labels, test_encodings, test_labels = tokenize_encode_data(train, test, 'text',
                                                                                          'category', checkpoint)
        validation_data = (test_encodings, np.array(test_labels))

        history, fitted_model = transformer_model_fit(compiled_model, train_encodings, train_labels,
                                                      test_encodings, test_labels, epochs=50)

        Big_Matrix.loc[iexp, 'num_categ'] = ilab
        Big_Matrix.loc[iexp, 'sample_size'] = i_ninst
        Big_Matrix.loc[iexp, 'val_f1'] = np.max(history.history['val_f1'])

## Record the results
Big_Matrix.to_csv('Big_Matrix32-64.csv',index=False)