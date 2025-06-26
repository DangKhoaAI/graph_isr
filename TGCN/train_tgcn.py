import logging
import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config, create_arg_parser
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation


def run(configs, args):
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Override config paths with command line arguments if provided
    root_dir = args.root if args.root else configs.root_dir
    
    # Build paths
    split_file = os.path.join(configs.splits_dir, f'{args.subset}.json')
    pose_data_root = configs.pose_data_root
    
    # Create output directories
    os.makedirs(configs.output_dir, exist_ok=True)
    os.makedirs(os.path.join(configs.checkpoints_dir, args.subset), exist_ok=True)
    
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples,
                                 features_dir=configs.features_dir)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='k_copies',
                               features_dir=configs.features_dir)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    # setup the model
    model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cuda()

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    for epoch in range(int(epochs)):
        # train, test model
        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=os.path.join(configs.checkpoints_dir, args.subset))

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        np.save(os.path.join(configs.output_dir, 'epoch_training_losses.npy'), np.array(epoch_train_losses))
        np.save(os.path.join(configs.output_dir, 'epoch_training_scores.npy'), np.array(epoch_train_scores))
        np.save(os.path.join(configs.output_dir, 'epoch_test_loss.npy'), np.array(epoch_val_losses))
        np.save(os.path.join(configs.output_dir, 'epoch_test_score.npy'), np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(), os.path.join(configs.checkpoints_dir, args.subset, 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))

    utils.plot_curves(output_dir=configs.output_dir)

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to=os.path.join(configs.output_dir, 'train-conf-mat'))
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, 
                                save_to=os.path.join(configs.output_dir, 'val-conf-mat'))


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Use subset-specific config if available, otherwise use default
    if os.path.exists(os.path.join('configs', f'{args.subset}.ini')):
        config_file = os.path.join('configs', f'{args.subset}.ini')
    else:
        config_file = args.config
        
    configs = Config(config_file)

    # Setup logging
    log_file = os.path.join(configs.output_dir, f'{args.subset}.log')
    os.makedirs(configs.output_dir, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(configs=configs, args=args)
    logging.info('Finished main.run()')
