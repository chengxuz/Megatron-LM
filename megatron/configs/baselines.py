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


def change_to_1d7b_from_1d3b(args):
    args.num_layers = 32
    args = name_keyword_change(
            args, new_name='1d7b')
    return args


def change_to_w1d7b_from_1d3b(args):
    args.kv_channels = 74
    args.hidden_size = args.kv_channels * 32
    args.ffn_hidden_size = args.hidden_size * 4
    args.micro_batch_size = 8
    args.global_batch_size = 64 # it was run as 64 at the beginning
    args = name_keyword_change(
            args, new_name='w1d7b')
    return args

def change_to_2d7b_from_1d3b(args):
    args.num_layers = 32
    args.hidden_size = 2560
    args.micro_batch_size = 8
    args.kv_channels = 2560 // 32
    args.ffn_hidden_size = 10240
    args = name_keyword_change(
            args, new_name='2d7b')
    return args


def change_to_att1d3b_from_1d3b(args):
    args = name_keyword_change(
            args, new_name='att1d3b')
    return args
