import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # In/Out
        self.parser.add_argument("--dataset_path", type=str, default="dataset")
        self.parser.add_argument('--output_dir', default='./output', help='output_directory')
        self.parser.add_argument('--output_pth', default='dict.pth', help='output pth filename')

        # training
        self.parser.add_argument("--epoch", type=int, default=10)
        self.parser.add_argument("--batch_size", type=int, default=1024)
        self.parser.add_argument("--learning_rate", type=float, default=0.01)
        self.parser.add_argument("--random_seed", type=int, default=0, help="random seed")

        # DDP
        self.parser.add_argument("--local_rank", type=int, help="local gpu id")

    def parse(self):
        return self.parser.parse_args()