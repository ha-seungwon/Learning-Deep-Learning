import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.007,
                    help='base learning rate')

parser.add_argument('--model', type=str, default="deeplabb3_gcn",
                    help='model name')

args = parser.parse_args()
