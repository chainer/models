# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
from datetime import datetime

import chainer
import logzero
from logzero import logger


class Resource(object):
    """
    Helper class for the experiment.
    """

    def __init__(self, args, train=True):
        self.args = args  # argparse object
        self.logger = logger
        self.start_time = datetime.today()
        self.config = None  # only used for the inference

        if train:  # for training
            self.output_dir = self._return_output_dir()
            self.create_output_dir()
            log_filename = 'train.log'
        else:  # for inference
            self.output_dir = os.path.dirname(args.model)
            self.model_name = os.path.basename(args.model)
            log_filename = 'inference_{}.log'.format(self.model_name)

        log_name = os.path.join(self.output_dir, log_filename)
        logzero.logfile(log_name)
        self.log_name = log_name
        self.logger.info('Log filename: [{}]'.format(log_name))

    def _return_output_dir(self):
        dir_name = '{}_{}_seed_{}_optim_{}_tau_{}_batch_{}_M_{}_K_{}'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S'),
            self.args.dir_prefix,
            self.args.seed,
            self.args.optimizer,
            self.args.tau,
            self.args.batchsize,
            self.args.n_codebooks,
            self.args.n_centroids
        )
        output_dir = os.path.abspath(os.path.join(self.args.out, dir_name))
        return output_dir

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info('Output Dir is created at [{}]'.format(self.output_dir))
        else:
            self.logger.info('Output Dir [{}] already exists'.format(self.output_dir))

    def dump_git_info(self):
        """
        returns git commit id, diffs from the latest commit
        """
        if os.system('git rev-parse 2> /dev/null > /dev/null') == 0:
            self.logger.info('Git repository is found. Dumping logs & diffs...')
            git_log = '\n'.join(
                l for l in
                subprocess.check_output('git log --pretty=fuller | head -7', shell=True).decode('utf8').split('\n') if
                l)
            self.logger.info(git_log)

            git_diff = subprocess.check_output('git diff', shell=True).decode('utf8')
            self.logger.info(git_diff)
        else:
            self.logger.warn('Git repository is not found. Continue...')

    def dump_command_info(self):
        """
        returns command line arguments / command path / name of the node
        """
        self.logger.info('Command name: {}'.format(' '.join(sys.argv)))
        self.logger.info('Command is executed at: [{}]'.format(os.getcwd()))
        self.logger.info('Program is running at: [{}]'.format(os.uname().nodename))

    def dump_chainer_info(self):
        """
        returns chainer, CuPy and CuDNN version info
        """
        chainer.print_runtime_info()

    def dump_python_info(self):
        """
        returns python version info
        """
        self.logger.info('Python Version: [{}]'.format(sys.version.replace('\n', '')))

    def save_config_file(self):
        """
        save argparse object into config.json
        config.json is used during the inference
        """
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as fo:
            dumped_config = json.dumps(vars(self.args), sort_keys=True, indent=4)
            fo.write(dumped_config)
            self.logger.info('HyperParameters: {}'.format(dumped_config))

    def dump_duration(self):
        end_time = datetime.today()
        self.logger.info('EXIT TIME: {}'.format(end_time.strftime('%Y%m%d - %H:%M:%S')))
        duration = end_time - self.start_time
        logger.info('Duration: {}'.format(str(duration)))
        logger.info('Remember: log is saved in {}'.format(self.output_dir))

    def load_config(self):
        """
        load config.json and recover hyperparameters that are used during the training
        """
        config_path = os.path.join(self.output_dir, 'config.json')
        self.config = json.load(open(config_path, 'r'))
        self.logger.info('Loaded config from {}'.format(config_path))
