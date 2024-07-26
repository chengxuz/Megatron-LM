import os
import sys
import pickle
import ipdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,
                                             os.path.pardir)))

from megatron.training import get_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import initialize_megatron
from megatron.training import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.core.models.gpt.model_providers import model_provider

from tasks.use_lm_eval import get_tasks_args
import torch
import numpy as np


def main():
    initialize_megatron(
            extra_args_provider=get_tasks_args,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            )

    args = get_args()
    args.return_qk = True
    set_args(args)

    teacher_model = get_model(
            model_provider,
            wrap_with_ddp=False)
    teacher_model = teacher_model[0]
    tokenizer = get_tokenizer()

    args.attention_copy = True
    args.return_qk = False
    args.kv_channels = 32
    args.hidden_size = args.kv_channels * 32
    args.ffn_hidden_size = args.hidden_size * 4
    args.att_sub_method = 'layer_random_forth_6'
    set_args(args)

    model = get_model(
            model_provider,
            wrap_with_ddp=False)
    model = model[0]
    model.module.teacher_model = teacher_model

    input_tokens = tokenizer.tokenize('I am a cat walking on a ')
    input_tokens = torch.from_numpy(np.asarray([input_tokens])).cuda()
    #ipdb.set_trace()
    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(
                input_tokens)
        model_outputs = model(
                input_tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                )
    ipdb.set_trace()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
