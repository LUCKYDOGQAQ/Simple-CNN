import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--data_dir', default='../data/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='../logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        # train args
        parser.add_argument('--train_epochs', default=15, type=int,
                            help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
