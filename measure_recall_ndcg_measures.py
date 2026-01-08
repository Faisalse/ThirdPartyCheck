import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="lightgcn", help="model/framework_name: lightgcn/ngcf")
args = parser.parse_args()
print(args.name)

if args.name == "lightgcn":
    dataset_names = ["gowalla", "amazon-book", "yelp"]