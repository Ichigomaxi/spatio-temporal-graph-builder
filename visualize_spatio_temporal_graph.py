import argparse
# from datasets.NuscenesDataset import NuscenesDataset
from visualization.visualize_graph import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--mode', dest='mode', type=str, help='mode of visualization single or nothing')
    args = parser.parse_args()
    print('Arguments Passed:\n',args.mode)

    main(args.mode)