import datetime
from program_constants import *

ret = '\n'
spacer = '\t'

class ConfigLogger:
    def __init__(self):
        now = datetime.datetime.now()
        self.timestamp = now.strftime("%H:%M %d %b")
        self.filename = now.strftime("%d-%H_%M_%S")
        self.acc_train = self.prec_train = self.rec_train = self.f1_train = 0
        self.acc_validation = self.prec_validation = self.rec_validation = self.f1_validation = 0
        self.acc_test = self.pre_test = self.rec_test = self.f1_test = 0
        self.model_description = ''

    def set_train_score(self, acc, pre, rec, f1):
        self.acc_train = acc
        self.pre_train = pre
        self.rec_train = rec
        self.f1_train = f1
        
    def set_validation_score(self, acc, pre, rec, f1):
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
                '| LOG {}|'.format(self.timestamp), ret,
                '=================', ret, ret,
                'Model:', ret,
                self.model_description, ret, ret,
                'Data bal.: ', spacer, str(BALANCE_DATA), ret,
                'Pix depth: ', spacer, str(PIXEL_DEPTH), ret,
                'Num img: ', spacer, str(NUM_IMAGES), ret,
                'Train size: ', spacer, str(TRAINING_SIZE), ret,
                'Val size: ', spacer, str(VALIDATION_SIZE), ret,
                'Test size: ', spacer, str(TEST_SIZE), ret,
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

        with open("logs/{}.txt".format(self.filename), "w") as text_file:
            print(s, file=text_file)

    def get_timestamp(self):
        return self.filename

if __name__ == '__main__':
    logger = ConfigLogger()
    logger.save()