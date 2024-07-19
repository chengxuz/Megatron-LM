import os


def change_to_1d7b_from_1d3b(args):
    args.num_layers = 32
    args.save = args.save.replace('1d3b', '1d7b')
    args.load = args.load.replace('1d3b', '1d7b')
    os.system(f'mkdir -p {args.save}')
    args.tensorboard_dir = args.tensorboard_dir.replace('1d3b', '1d7b')
    os.system(f'mkdir -p {args.tensorboard_dir}')
    return args
