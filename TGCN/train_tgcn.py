import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from configs import create_arg_parser, get_config_from_args
from tgcn_model import GCN_muti_att
from lmdb_sign_dataset import LMDBSignDataset
from train_utils import train, validation


def safe_collate(batch):
    xs, ys, *_ = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.tensor(ys)
    return xs, ys

def run(configs, args):
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    split_file = os.path.join(configs.splits_dir, f'{args.subset}.json')
    lmdb_path = configs.lmdb_path
    
    # Create output directories
    os.makedirs(configs.output_dir, exist_ok=True)
    os.makedirs(os.path.join(configs.checkpoints_dir, args.subset), exist_ok=True)
    
    # setup dataset
    train_dataset = LMDBSignDataset(index_file_path=split_file, split=['train', 'val'], lmdb_path=lmdb_path,
                                 num_samples=configs.num_samples, sample_strategy='rnd_start', return_video_id=False)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, collate_fn=safe_collate)

    val_dataset = LMDBSignDataset(index_file_path=split_file, split='test', lmdb_path=lmdb_path,
                               num_samples=configs.num_samples, sample_strategy='k_copies', return_video_id=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    # setup the model
    model = GCN_muti_att(input_feature=configs.num_samples*2, hidden_feature=configs.hidden_size,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=configs.drop_p, num_stage=configs.num_stages).cuda()
    # setup training parameters, learning rate, optimizer, scheduler
    optimizer = optim.Adam(model.parameters(), lr=configs.init_lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    for epoch in range(int(configs.max_epochs)):
        # train, test model
        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(configs.log_interval, model,
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    args = parser.parse_args() 

    configs = get_config_from_args(args)

    # Setup logging
    log_file = os.path.join(configs.output_dir, f'{args.subset}.log')
    os.makedirs(configs.output_dir, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(configs=configs, args=args)
    logging.info('Finished main.run()')
