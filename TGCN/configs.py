import configparser
import argparse
import os


class Config:
    def __init__(self, config_path, args):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)

        paths_config = config['PATHS']

        # paths
        # Prioritize command-line args, then fall back to config file.
        self.root_dir = args.root_dir if args.root_dir is not None else paths_config.get('ROOT_DIR')
        self.pose_data_root = args.pose_data_root if args.pose_data_root is not None else paths_config.get('POSE_DATA_ROOT')
        self.splits_dir = args.splits_root if args.splits_root is not None else paths_config.get('SPLITS_ROOT')

        # Ensure essential paths are defined
        if self.root_dir is None:
            raise ValueError("ROOT_DIR must be provided via --root_dir argument or in the config file.")
        if self.pose_data_root is None:
            raise ValueError("POSE_DATA_ROOT must be provided via --pose_data_root argument or in the config file.")
        if self.splits_dir is None:
            raise ValueError("SPLITS_ROOT must be provided via --splits_root argument or in the config file.")

        # Paths from config file, with manual interpolation
        self.features_dir = paths_config['FEATURES_DIR'] % {'ROOT_DIR': self.root_dir}
        self.checkpoints_dir = paths_config['CHECKPOINTS_DIR']
        self.output_dir = paths_config['OUTPUT_DIR']
        self.config_dir = paths_config['CONFIG_DIR'] % {'ROOT_DIR': self.root_dir}
        self.archive_dir = paths_config['ARCHIVE_DIR'] % {'ROOT_DIR': self.root_dir}
        self.lmdb_path = paths_config['LMDB_PATH'] % {'ROOT_DIR': self.root_dir}

        # training
        train_config = config['TRAIN']
        # Prioritize command-line args, then fall back to config file, then use hardcoded defaults
        self.batch_size = args.batch_size if args.batch_size is not None else int(train_config.get('BATCH_SIZE', 64))
        self.max_epochs = args.max_epochs if args.max_epochs is not None else int(train_config.get('MAX_EPOCHS', 200))

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
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Root directory path. Overrides value from config file.')
    parser.add_argument('--pose_data_root', type=str, default=None,
                        help='Pose data root directory path. Overrides value from config file.')
    parser.add_argument('--splits_root', type=str, default=None,
                        help='Splits root directory path. Overrides value from config file.')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of training epochs. Overrides value from config file.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training. Overrides value from config file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint filename (overrides config)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    return parser


def get_config_from_args(args):
    """
    Resolves the config file path from arguments and returns a Config object.
    It prioritizes a user-specified config file via the --config argument.
    If --config is not specified, it then prioritizes a subset-specific config file
    if it exists in the 'configs' directory, before falling back to the default.
    """
    # The default value for --config is 'config.ini'
    # If the user provides a different value, we prioritize it.
    if args.config != 'config.ini':
        config_file = args.config
    else:
        # If the user did not specify a custom config, use the subset-based logic
        subset_config_path = os.path.join('configs', f'{args.subset}.ini')
        if os.path.exists(subset_config_path):
            config_file = subset_config_path
        else:
            config_file = args.config  # Fallback to the default 'config.ini'

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    print(f"Loading configuration from: {config_file}")
    return Config(config_file, args)


if __name__ == '__main__':
    config_path = '/mnt/data/Work/Project/RESEARCH/Handsign-code/WLASL/TGCN/config.ini'
    print(str(Config(config_path)))