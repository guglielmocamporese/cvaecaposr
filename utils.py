##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os


def get_callbacks(args):
    callbacks = []

    # Model checkpoint
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor='validation_acc',
        save_last=True,
        mode='max',
    )
    model_checkpoint_clbk.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks += [model_checkpoint_clbk]
    return callbacks

def get_logger(args):
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'tmp'),
        name=args.dataset,
    )
    return tb_logger
