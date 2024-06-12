"""
Model Evaluation for itmAFA
"""
import os
import argparse
import logging
from lib import evaluation
import arguments


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/tmp/data/coco')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        # put the MSCOCO checkpoints here
        weights_bases = [
                        #'./models/itmAFA_coco',
        		        ]
    elif opt.dataset == 'f30k':
        # put the Flickr30K checkpoints here
        weights_bases = [
                        #'./models/itmAFA_f30k',
			            ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        arguments.seed_everything(1001) # set the random seeds for result reproduction
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            print(f"Currently, we are processing the COCO 5-fold 1K")
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
            # Evaluate COCO 5K
            print(f"Currently, we are processing the COCO 5K")
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)

        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)

if __name__ == '__main__':
    main()
