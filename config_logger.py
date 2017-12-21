import datetime
from pathlib import Path

import numpy as np

from program_constants import *

ret = '\n'
spacer = '\t'


class ConfigLogger:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.curr_path_str = "logs/{}".format(self.timestamp.strftime("%d-%H_%M_%S") + '/')
        curr_path = Path(self.curr_path_str)
        if not curr_path.is_dir():
            curr_path.mkdir()

        self.acc_train = self.prec_train = self.rec_train = self.f1_train = 0
        self.acc_validation = self.prec_validation = self.rec_validation = self.f1_validation = 0
        self.acc_test = self.pre_test = self.rec_test = self.f1_test = 0
        self.model_description = ''
        self.train_scores = []
        self.validation_scores = []
        self.iteration = 0

    def set_train_score(self, acc, pre, rec, f1):
        self.train_scores.append([acc, pre, rec, f1])
        self.acc_train = acc
        self.pre_train = pre
        self.rec_train = rec
        self.f1_train = f1

    def set_validation_score(self, acc, pre, rec, f1):
        self.validation_scores.append([acc, pre, rec, f1])
        self.acc_validation = acc
        self.pre_validation = pre
        self.rec_validation = rec
        self.f1_validation = f1

    def set_test_score(self, acc, pre, rec, f1):
        self.acc_test = acc
        self.pre_test = pre
        self.rec_test = rec
        self.f1_test = f1

    def describe_model(self, description):
        self.model_description = description

    def save(self):
        s = ''
        s = s.join(
            [
                '=================', ret,
                '| LOG {}|'.format(self.timestamp.strftime("%H:%M %d %b")), ret,
                '=================', ret, ret,
                'Model:', ret,
                self.model_description, ret, ret,
                'Data bal.: ', spacer, str(BALANCE_TRAIN_DATA), ret,
                'Data shfl.: ', spacer, str(SHUFFLE_DATA), ret,
                'Pix depth: ', spacer, str(PIXEL_DEPTH), ret,
                'Train size: ', spacer, str(TRAINING_PROP), ret,
                'Val size: ', spacer, str(VALIDATION_PROP), ret,
                'Test size: ', spacer, str(TEST_PROP), ret,
                'Seed num: ', spacer, str(SEED), ret,
                'Batch size: ', spacer, str(BATCH_SIZE), ret,
                'Patch size: ', spacer, str(IMG_PATCH_SIZE), ret,
                'Border size: ', spacer, str(PATCH_CONTEXT_SIZE), ret,
                'Adam rate: ', spacer, str(ADAM_INITIAL_LEARNING_RATE), ret,
                'Num epochs: ', spacer, str(NUM_EPOCHS), ret, ret,
                'Train scores', ret,
                '=================', ret,
                'Accuracy: ', spacer, str(self.acc_train), ret,
                'Precision: ', spacer, str(self.pre_train), ret,
                'Recall: ', spacer, str(self.rec_train), ret,
                'F1-score: ', spacer, str(self.f1_train), ret, ret,
                'Validation scores', ret,
                '=================', ret,
                'Accuracy: ', spacer, str(self.acc_validation), ret,
                'Precision: ', spacer, str(self.pre_validation), ret,
                'Recall: ', spacer, str(self.rec_validation), ret,
                'F1-score: ', spacer, str(self.f1_validation), ret, ret,
                'Test scores', ret,
                '=================', ret,
                'Accuracy: ', spacer, str(self.acc_test), ret,
                'Precision: ', spacer, str(self.pre_test), ret,
                'Recall: ', spacer, str(self.rec_test), ret,
                'F1-score: ', spacer, str(self.f1_test), ret
            ]
        )

        with open(self.curr_path_str + "training_configuration.txt", "w") as text_file:
            print(s, file=text_file)

        with open(self.curr_path_str + "train_scores.csv", "wb") as train_scores_file:
            np.savetxt(train_scores_file, self.train_scores, delimiter=',', fmt='%s',
                       header='accuracy, precision, recall, F1-score')
        with open(self.curr_path_str + "val_scores.csv", "wb") as validation_scores_file:
            np.savetxt(validation_scores_file, self.validation_scores, delimiter=',', fmt='%s',
                       header='accuracy, precision, recall, F1-score')

    def get_timestamp(self):
        return self.timestamp.strftime("%d-%H_%M_%S")


if __name__ == '__main__':
    logger = ConfigLogger()
    logger.save()
