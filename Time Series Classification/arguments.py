import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--model_name', type=str, default="LSTM",
                    help='model name')

args = parser.parse_args()