from .useful_funcs import name_keyword_change, change_to_1d3bhlf
from . import useful_funcs


def change_to_5B(args):
    args.data_path = [
            '/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_41-of-01650_text_document']
    args.train_iters = 120000
    args.lr_decay_iters = 100000

    old_name = 'gpt_test_train'
    new_name = 'gpt_train_5B'
    args.save = args.save.replace(old_name, new_name)
    args.load = args.load.replace(old_name, new_name)
    args.tensorboard_dir = args.tensorboard_dir.replace(old_name, new_name)
    return args


def base_1d3b(args):
    args = change_to_5B(args)
    args = name_keyword_change(
            args, new_name='1d3bdcy')
    return args

def base_2d4b(args):
    args = change_to_5B(args)
    args = useful_funcs.change_to_2d7b(args)
    args.num_layers = 29
    args = name_keyword_change(
            args, new_name='2d4bdcy')
    return args

def base_1d7b(args):
    args = change_to_5B(args)
    args.num_layers = 32
    args = name_keyword_change(
            args, new_name='1d7bdcy')
    return args


def base_1d3b_hlr(args):
    args = change_to_5B(args)
    args.lr = 2e-4
    args = name_keyword_change(
            args, new_name='1d3bdcy_hlr')
    return args
