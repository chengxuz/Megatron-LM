from .useful_funcs import name_keyword_change, change_to_1d3bhlf
from . import useful_funcs


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
    args = name_keyword_change(
            args, new_name='w1d7b_2')
    return args

def change_to_2d7b_from_1d3b(args):
    args = useful_funcs.change_to_2d7b(args)
    args = name_keyword_change(
            args, new_name='2d7b')
    return args

def change_to_2d7bftn_from_1d3b(args):
    args = useful_funcs.change_to_2d7b(args)

    # need to turn off the finetune after starting
    #args.finetune = True
    #args.pretrained_checkpoint = "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_2d7b/ckpts"
    args = name_keyword_change(
            args, new_name='2d7bftn')
    return args


def change_to_att1d3b_from_1d3b(args):
    #args.kv_channels = 32
    #args.hidden_size = args.kv_channels * 32
    #args.ffn_hidden_size = args.hidden_size * 4

    args.att_sub_method = 'layer_random_forth_6'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'
    args = name_keyword_change(
            args, new_name='att1d3b')
    return args


def change_to_att1d3bctl_from_1d3b(args):
    #args.att_sub_method = 'layer_random_forth_6'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'
    args = name_keyword_change(
            args, new_name='att1d3bctl')
    return args


def change_to_att1d3bctltrn_from_1d3b(args):
    args.attention_copy = True
    args.attention_teacher = '1d3b'
    args.micro_batch_size = 8
    args = name_keyword_change(
            args, new_name='att1d3bctltrn')
    return args


def change_to_att1d3bhlf_from_1d3b(args):
    args = change_to_1d3bhlf(args)
    args.att_sub_method = 'layer_random_forth_6'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'
    args = name_keyword_change(
            args, new_name='att1d3bhlf')
    return args


def change_to_att1d3bhlftrn_from_1d3b(args):
    args = change_to_att1d3bhlf_from_1d3b(args)
    args.attention_teacher = '1d3b'
    args = name_keyword_change(
            args, new_name='att1d3bhlftrn',
            old_name='att1d3bhlf')
    return args

def change_to_att1d3bhlfsch_from_1d3b(args):
    args = change_to_att1d3bhlf_from_1d3b(args)
    args.attention_teacher = '1d3b_scratch'
    args = name_keyword_change(
            args, new_name='att1d3bhlfsch',
            old_name='att1d3bhlf')
    return args
