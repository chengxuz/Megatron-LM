from .useful_funcs import name_keyword_change, change_to_1d3bhlf


def change_to_sw150M(args):
    args.data_path = ['/om2/group/evlab/llm_dataset/Megatron_datasets/sw_150M_gpt2_text_document']
    args.train_iters = 300000
    args.micro_batch_size = 32
    args.seq_length = 128
    args.save_interval = 20000
    return args


def att1d3b(args):
    args.att_sub_method = 'layer_random_forth_6'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'

    args = change_to_sw150M(args)
    args = name_keyword_change(
            args, new_name='sw_att1d3b')
    return args

def att1d3brnd(args):
    args.att_sub_method = 'layer_random'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'

    args = change_to_sw150M(args)
    args = name_keyword_change(
            args, new_name='sw_att1d3brnd')
    return args


def att1d3bctl(args):
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'
    args = change_to_sw150M(args)
    args = name_keyword_change(
            args, new_name='sw_att1d3bctl')
    return args


def att1d3bhlf(args):
    args = change_to_1d3bhlf(args)
    args.att_sub_method = 'layer_random_forth_6'
    args.attention_copy = True
    args.attention_teacher = '1d3b_frozen'

    args = change_to_sw150M(args)
    args = name_keyword_change(
            args, new_name='sw_att1d3bhlf')
    return args


def att1d3btrn(args):
    args.attention_copy = True
    args.attention_teacher = '1d3b'
    args = change_to_sw150M(args)
    args = name_keyword_change(
            args, new_name='sw_att1d3btrn')
    return args
