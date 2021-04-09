##################################################
# Imports
##################################################

import json
import pytorch_lightning as pl

# Custom
from config import parse_args
from dataloader import get_dataloaders
from models import cvaecaposr
from utils import get_logger, get_callbacks


# Main function
def main(args):

    # Dataloaders
    dls, data_info = get_dataloaders(args)

    # Model
    model = cvaecaposr.get_model(args, data_info)

    # Callbacks and logger
    callbacks = get_callbacks(args)
    tb_logger = get_logger(args)

    # Trainer
    if args.mode in ['train', 'training']:
        trainer = pl.Trainer(
            max_epochs=30, 
            gpus=1, 
            callbacks=callbacks,
            num_sanity_val_steps=0,
            logger=tb_logger,
        )
        
        # Fit
        trainer.fit(
            model, 
            train_dataloader=dls['known']['train_aug'], 
            val_dataloaders=dls['known']['validation'],
        )
        
        # Test (loading best model)
        trainer.test(model=None, test_dataloaders=dls['test'])
        
    elif args.mode in ['test', 'testing']:
        trainer = pl.Trainer( 
            gpus=1, 
            callbacks=callbacks,
            logger=tb_logger,
        )
        
        # Test
        trainer.test(model=model, test_dataloaders=dls['test'])
    else:
        raise Exception(f'Error. Mode "{args.mode}" is not supported.')



##################################################
# Main
##################################################

if __name__ == '__main__':

    # Parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    # Main
    main(args)
