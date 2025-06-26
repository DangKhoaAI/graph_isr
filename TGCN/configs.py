import configparser
import argparse
import os


class Config:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        # paths
        paths_config = config['PATHS']
        self.root_dir = paths_config['ROOT_DIR']
        self.data_dir = paths_config['DATA_DIR']
        self.pose_data_root = paths_config['POSE_DATA_ROOT']
        self.splits_dir = paths_config['SPLITS_DIR']
        self.features_dir = paths_config['FEATURES_DIR']
        self.checkpoints_dir = paths_config['CHECKPOINTS_DIR']
        self.output_dir = paths_config['OUTPUT_DIR']
        self.config_dir = paths_config['CONFIG_DIR']
        self.archive_dir = paths_config['ARCHIVE_DIR']
        self.lmdb_path = paths_config['LMDB_PATH']

        # training
        train_config = config['TRAIN']
        self.batch_size = int(train_config['BATCH_SIZE'])
        self.max_epochs = int(train_config['MAX_EPOCHS'])
        self.log_interval = int(train_config['LOG_INTERVAL'])
        self.num_samples = int(train_config['NUM_SAMPLES'])
        self.drop_p = float(train_config['DROP_P'])

        # optimizer
        opt_config = config['OPTIMIZER']
        self.init_lr = float(opt_config['INIT_LR'])
        self.adam_eps = float(opt_config['ADAM_EPS'])
        self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

        # GCN
        gcn_config = config['GCN']
        self.hidden_size = int(gcn_config['HIDDEN_SIZE'])
        self.num_stages = int(gcn_config['NUM_STAGES'])

        # model
        model_config = config['MODEL']
        self.checkpoint_name = model_config['CHECKPOINT_NAME']

    def __str__(self):
        return 'bs={}_ns={}_drop={}_lr={}_eps={}_wd={}'.format(
            self.batch_size, self.num_samples, self.drop_p, self.init_lr, self.adam_eps, self.adam_weight_decay
        )


def create_arg_parser():
    """Create argument parser for command line arguments"""
    parser = argparse.ArgumentParser(description='TGCN Training/Testing')
    parser.add_argument('--config', type=str, default='config.ini',
                        help='Path to configuration file')
    parser.add_argument('--subset', type=str, default='asl100',
                        help='Dataset subset (asl100, asl300, asl1000, asl2000)')
    parser.add_argument('--root', type=str, default=None,
                        help='Root directory path (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint filename (overrides config)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    return parser


if __name__ == '__main__':
    config_path = '/mnt/data/Work/Project/RESEARCH/Handsign-code/WLASL/TGCN/config.ini'
    print(str(Config(config_path)))