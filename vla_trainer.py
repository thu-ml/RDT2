# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""


import time
from typing import TYPE_CHECKING, List, Optional, Union

# isort: on

import numpy as np
import torch
from packaging import version
from torch.utils.data import DataLoader, Dataset

from transformers import __version__
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    find_batch_size,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_datasets_available,
    is_in_notebook,
    is_torch_xla_available,
)
from transformers.integrations.tpu import tpu_spmd_dataloader

from transformers.trainer import logger, Trainer

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False



if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets


class VLATrainer(Trainer):

    def __init__(
        self,
        num_eval_datasets: int = 2, 
        num_eval_batches: int = 4,
        use_default_collate_fn_for_eval: bool = False,
        *args, 
        **kwargs
    ):
        """Initializes VLATrainer.

        Args:
            num_eval_datasets (int, optional): Number of evaluation datasets to use. Defaults to 3.
            num_eval_batches (int, optional): Number of batches to evaluate for each dataset. Defaults to 10.
        """
        super().__init__(*args, **kwargs)
        
        self.num_eval_datasets = num_eval_datasets
        self.num_eval_batches = num_eval_batches
        self.use_default_collate_fn_for_eval = use_default_collate_fn_for_eval

        # initialize the index of the evaluation dataset
        # we only evaluate `num_eval_datasets` datasets in a round-robin manner
        self.eval_dataset_index = 0
        self.eval_dataset_names = None
        if isinstance(self.eval_dataset, dict):
            self.eval_dataset_names = sorted(self.eval_dataset.keys())

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        collate_fn = (
            None if self.use_default_collate_fn_for_eval 
            else lambda examples: examples
        )
        # NOTE: we use customized collate_fn for evaluation
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": True # shuffle test dataloader every time
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        containers = {
            "action_valid_rate": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error_pos": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_geodesic_error_rot": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error_width": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
        }

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            if step >= self.num_eval_batches:
                break

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                metrics_per_step = self.compute_metrics(
                    model=model, examples=inputs
                )

            if is_torch_xla_available():
                xm.mark_step()

            for key, value in metrics_per_step.items():
                # Update containers
                gathered_values = self.gather_function(value.repeat(batch_size))
                containers[key].add(gathered_values.detach())

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                for key in containers:
                    containers[key].to_cpu_and_numpy()

                del metrics_per_step
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        for key in containers:
            containers[key] = containers[key].get_arrays()

        # FIXME(lingxuan): it's not right for there may be invalid prediction
        # but there's almost no invalid prediction even during the early phase of training
        # so we do not consider it for now
        num_samples = observed_num_examples

        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        for metric_key, value in containers.items():
            if isinstance(value, list) and value:
                metrics[f"{metric_key_prefix}_{metric_key}"] = np.concatenate(value).mean().item()
            elif isinstance(value, np.ndarray):
                metrics[f"{metric_key_prefix}_{metric_key}"] = value.mean().item()

        if hasattr(self, "jit_compilation_time" ):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        We only evaluate `num_eval_datasets` datasets in a round-robin manner.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if (
            isinstance(eval_dataset, dict) and 
            len(eval_dataset) > self.num_eval_datasets
        ):
            # use round-robin strategy to select `num_eval_datasets` evaluation datasets
            total_eval_datasets = len(self.eval_dataset_names)
            next_eval_dataset_names = [
                self.eval_dataset_names[
                    (self.eval_dataset_index + i) % total_eval_datasets
                ]
                for i in range(self.num_eval_datasets)
            ]
            
            # select designated evaluation datasets
            eval_dataset = {
                eval_dataset_name: eval_dataset[eval_dataset_name]
                for eval_dataset_name in next_eval_dataset_names
            }
            # maintain the index of the evaluation dataset
            self.eval_dataset_index = (
                (self.eval_dataset_index + self.num_eval_datasets) 
                    % total_eval_datasets
            )
        
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
