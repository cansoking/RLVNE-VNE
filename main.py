import event
import argparse


sub_path = "sub.txt"
req_path = "r-2000-50-50-20-10-5-25"
num_req = 1000
start = 0
num_epoch = 200
parser = argparse.ArgumentParser(description='params')
parser.add_argument('-model', type=str, default='conv')
parser.add_argument('-test', type=int, default=0)
parser.add_argument('-rate', type=float, default=0.005)
args = parser.parse_args()

# event.run_base(sub_path, req_path, num_req, start)
    
event.run_rl(sub_path, req_path, num_req, start, num_epoch, target='rev', test=args.test, model_str=args.model, learning_rate=args.rate)
