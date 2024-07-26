import os
import torch
from functools import partial
import pickle
import importlib

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel, AttCopyGPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_att_copy_gpt_layer_with_transformer_engine_spec,
)
from megatron.training.global_vars import set_args


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    attention_copy = getattr(args, 'attention_copy', False)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                if not attention_copy:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
                else:
                    transformer_layer_spec = get_att_copy_gpt_layer_with_transformer_engine_spec(
                            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        if not attention_copy:
            model_builder = GPTModel
        else:
            model_builder = AttCopyGPTModel
        model = model_builder(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

        if attention_copy and (getattr(args, 'attention_teacher', None) is not None):
            teacher_model = load_attention_teacher(args.attention_teacher)
            model.teacher_model = teacher_model
    return model


def load_attention_teacher(teacher_name):
    orig_args = get_args()
    megatron_meta_dir = '/om2/user/chengxuz/megatron_related'
    if teacher_name in [
            '1d3b_frozen', '1d3b',
            '1d3b_scratch']:
        teacher_args_path = os.path.join(
                megatron_meta_dir, 'gpt_test_train/gpt2_1d7b/typical_args.pkl')
        if teacher_name != '1d3b_scratch':
            load_step = 440000
        else:
            load_step = None
    teacher_args = pickle.load(open(teacher_args_path, 'rb'))
    teacher_args.return_qk = True
    set_args(teacher_args)

    teacher_model = get_model(
            model_provider,
            wrap_with_ddp=False)
    if load_step is not None:
        teacher_args.iteration, teacher_args.num_floating_point_operations_so_far = load_checkpoint(
            teacher_model, optimizer=None, opt_param_scheduler=None,
            load_step=load_step)
    teacher_model = teacher_model[0].module
    if teacher_name in ['1d3b_frozen']:
        for param in teacher_model.parameters():
            param.requires_grad = False
    set_args(orig_args)
    return teacher_model
