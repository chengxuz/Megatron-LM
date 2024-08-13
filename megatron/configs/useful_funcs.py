import os


def name_keyword_change(
        args, new_name,
        old_name='1d3b'):
    args.save = args.save.replace(old_name, new_name)
    args.load = args.load.replace(old_name, new_name)
    os.system(f'mkdir -p {args.save}')
    args.tensorboard_dir = args.tensorboard_dir.replace(old_name, new_name)
    os.system(f'mkdir -p {args.tensorboard_dir}')
    return args


def change_to_1d3bhlf(args):
    args.kv_channels = 32
    args.hidden_size = args.kv_channels * 32
    args.ffn_hidden_size = args.hidden_size * 4
    return args


def change_to_2d7b(args):
    args.num_layers = 32
    args.hidden_size = 2560
    args.micro_batch_size = 8
    args.kv_channels = 2560 // 32
    args.ffn_hidden_size = 10240
    return args
