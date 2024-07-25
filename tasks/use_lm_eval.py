"""Evaluate the megatron models using lm-eval"""

import os
import sys
import pickle
import ipdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.training import print_rank_0, is_last_rank
from megatron.training import get_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import initialize_megatron
from megatron.training import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.training.checkpointing import load_checkpoint
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids

import lm_eval
from lm_eval.base import BaseLM
import torch

from lm_eval import tasks, evaluator

from lm_eval.tasks.winogrande import Winogrande
from lm_eval import tasks


class WinograndeTrain(Winogrande):
    def validation_docs(self):
        return self.dataset["train"]


tasks.TASK_REGISTRY['winogrande_train'] = WinograndeTrain
tasks.ALL_TASKS.append('winogrande_train')


def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        args = get_args()
        config = core_transformer_config_from_args(args)
        use_te = args.transformer_impl == "transformer_engine"

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
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

            model = GPTModel(
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

        return model
    return model_provider


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, default='winogrande_train',
                       help='Task name.')
    group.add_argument('--overwrite', default=False, action='store_true',
                       help='Task name.')
    group.add_argument(
            '--setting', 
            default=None, type=str, 
            action='store')
    return parser


class MegatronModelWrapper(BaseLM):
    def __init__(
            self, 
            model,
            tokenizer,
            device='',
            batch_size=8,
            extra_forward_mode=None,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.lm = model.to(self.device)
        self.lm.eval()

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            attention_mask, position_ids = _build_attention_mask_and_position_ids(
                    inps)
            #ipdb.set_trace()
            ret_logits = self.lm(
                inps,
                position_ids=position_ids,
                attention_mask=attention_mask,
                )[:, -inps.shape[1]:, :self.vocab_size]
            return ret_logits

    @property
    def max_length(self):
        try:
            max_len = self.lm.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            max_len = getattr(
                    self.lm.config,
                    'max_position_embeddings',
                    128)
        if max_len < 0:
            max_len = 10000000
        return max_len

    @property
    def eot_token_id(self):
        if self.tokenizer.eos_token_id is not None:
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id
        else:
            # GIT does not have eos
            return self.tokenizer.sep_token_id

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.tokenize(string)

    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError
        return self.lm.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


if __name__ == '__main__':

    initialize_megatron(
            extra_args_provider=get_tasks_args,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            )

    args = get_args()
    task_dict = tasks.get_task_dict([args.task])
    all_ckpts = os.listdir(args.load)
    all_ckpts = filter(
            lambda x: x.startswith('iter_'),
            all_ckpts)
    all_ckpts = [
            int(_name.split('_')[1])\
            for _name in all_ckpts]
    #print(all_ckpts)
    all_results = {}
    save_path = os.path.join(args.tensorboard_dir, f'{args.task}.pkl')
    if os.path.exists(save_path) and not args.overwrite:
        all_results = pickle.load(open(save_path, 'rb'))
    all_ckpts = list(
            filter(
                lambda x: x not in all_results,
                all_ckpts))

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(
            get_model_provider(),
            wrap_with_ddp=False)
    tokenizer = get_tokenizer()

    #all_ckpts = [20000,]
    print(all_ckpts)
    for _step in all_ckpts:
        if args.load is not None or args.pretrained_checkpoint is not None:
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model, optimizer=None, opt_param_scheduler=None,
                load_step=_step)

        model_wrapper = MegatronModelWrapper(
                model=model[0],
                tokenizer=tokenizer,
                #device='cpu',
                )
        results = evaluator.evaluate(
                lm=model_wrapper,
                task_dict=task_dict,
                num_fewshot=0,
                limit=None,
                bootstrap_iters=100000,
                description_dict=None,
                decontamination_ngrams_path=None,
                )
        print(_step, results)
        all_results[_step] = results
        pickle.dump(all_results, open(save_path, 'wb'))
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        set_args(args)
    torch.distributed.destroy_process_group()
