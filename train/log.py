from typing import Literal
import logging
import os
from datetime import datetime

START_EPOCH_STR = 'Starting epoch {0}'
END_EPOCH_STR = 'Epoch {0} ended. Train Loss: {1:1.3}. Validation Loss: {2:1.3}'
END_EPOCH_STR_NO_VAL = 'Epoch {0} ended. Train Loss: {1:1.3}'
SAVING_CHECKPOINT_STR = 'Saving checkpoint of epoch {0} at \"{1}\"'


"""
Logger for training
"""
class TrainLogger():
    def __init__(self,
                 log_name : str,
                 output_dir : str = "logs",
                 logging_level : int = logging.DEBUG) -> None:
        
        self.log_object = logging.getLogger(log_name)
        logging.basicConfig(filename=os.path.join(output_dir, datetime.now().strftime('%m-%d %H:%M:%S.log')),
                            encoding='utf-8',
                            format="[%(asctime)s]%(message)s",
                            datefmt="%H:%M:%S",
                            level=logging_level)

    def log_start_epoch(self, epoch : int) -> None:
        logging.info(START_EPOCH_STR.format(epoch))

    def log_end_epoch(self, epoch : int, train_loss : int, validation_loss : int = None):
        if validation_loss is None:
            logging.info(END_EPOCH_STR_NO_VAL.format(epoch, train_loss))
        else:
            logging.info(END_EPOCH_STR.format(epoch, train_loss, validation_loss))

    def log_save(self, epoch : int, path : str):
        logging.info(SAVING_CHECKPOINT_STR.format(epoch, path))

    def log(self, message : str):
        logging.info(message)