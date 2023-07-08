import os

import pprint

from tqdm import tqdm, trange
import numpy as np
import mlxu
import gcsfs

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)

import transformers
from transformers import AutoTokenizer
import torch
from datasets import load_from_disk, concatenate_datasets
from dataclasses import dataclass


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    batch_size=1,
    preprocessing_num_workers=1,
    seq_length=2048,
    tokenizer_path="tokenizer",
    data_path="data",
    output_dir="output",
    bucket_project_name='',
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp16',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=1000,
    save_model_freq=1000,
    save_milestone_freq=1000,
    repeat_corpus=1,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def make_inputs(
    batch_data, tokenizer, max_length, im_start_token, im_end_token
):
    
    input_ids = [
        datum["input_ids"][:max_length] + [tokenizer.pad_token_id] * (max_length - len(datum["input_ids"][:max_length]))
        for datum in batch_data
    ]
    input_ids = np.array(input_ids, dtype=np.int32)
    positions = [datum["positions"] for datum in batch_data]

    batch_size = len(batch_data)
    attention_mask = np.zeros((batch_size, max_length, max_length), dtype=np.uint8)
    position_ids = np.zeros((batch_size, max_length), dtype=np.int32)
    target_tokens = np.zeros((batch_size, max_length), dtype=np.int32)
    target_tokens.fill(tokenizer.eos_token_id)
    loss_mask = np.ones((batch_size, max_length), dtype=np.uint8)
    
    for i, position in enumerate(positions):
        start = 0
        for length in position:
            if start >= max_length: continue
            length = min(length, max_length)
            end = start + length
            mask = np.tril(np.ones((length, length), dtype=np.uint8))
            attention_mask[i, start:end, start:end] = mask
            position_ids[i, start:end] = np.arange(length)
            target_tokens[i, start:end - 1] = input_ids[i, start + 1:end]
            start = end
        loss_mask[i, start:] = 0
    
    target_tokens[target_tokens == im_end_token] = tokenizer.eos_token_id
    loss_mask[input_ids == im_start_token] = 0
    loss_mask[input_ids == im_end_token] = 0
        
    batch = {
        'input_tokens': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'target_tokens': target_tokens,
        'loss_masks': loss_mask,
    }
    return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    max_length: int
    im_start_token: int
    im_end_token: int

    def __call__(self, instances):
        features = make_inputs(
            instances, 
            self.tokenizer,
            max_length=self.max_length,
            im_start_token=self.im_start_token,
            im_end_token=self.im_end_token,
        )
        return features


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    # tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_path)
    
    try:
        im_start_token = tokenizer.convert_token_to_id("<|im_start|>")
        im_end_token = tokenizer.convert_token_to_id("<|im_end|>")
    except:
        im_start_token, im_end_token = 0, 1    
    
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, 
        max_length=FLAGS.seq_length,
        im_start_token=im_start_token,
        im_end_token=im_end_token,
    )
    
    fs = None
    if FLAGS.bucket_project_name != '':
        storage_options = {"project": FLAGS.bucket_project_name}
        fs = gcsfs.GCSFileSystem(**storage_options)
    datasets = []
    for _ in range(FLAGS.repeat_corpus):
        for data_path in FLAGS.data_path.split(','):
            if fs is not None:
                dataset = load_from_disk(data_path, storage_options=fs.storage_options)
            else:
                dataset = load_from_disk(data_path)
            datasets.append(dataset)
    datasets = concatenate_datasets(datasets)
    
    train_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=FLAGS.batch_size,
        shuffle=False,  # NOTE Data is already shuffled when serialized
        num_workers=FLAGS.preprocessing_num_workers,
        drop_last=True,
        collate_fn=data_collator,
        persistent_workers=True if FLAGS.preprocessing_num_workers > 0 else False,
    )
    
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
        
    # tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    # dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        # bos_token_id=dataset.tokenizer.bos_token_id,
        # eos_token_id=dataset.tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=FLAGS.seq_length
    ))
    if llama_config.vocab_size < tokenizer.vocab_size:
        llama_config.update(dict(vocab_size=tokenizer.vocab_size))

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((FLAGS.batch_size, FLAGS.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((FLAGS.batch_size, FLAGS.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((FLAGS.batch_size, FLAGS.seq_length, FLAGS.seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, FLAGS.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=None, # dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))
        print(f"Start at {start_step} steps...")

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        # for step, (batch, dataset_metrics) in zip(step_counter, dataset):
        num_tokens = 0
        for step, batch in zip(step_counter, train_loader):
            
            num_tokens += batch["loss_masks"].sum()
            if step < start_step:
                continue
            
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step, "num_tokens": num_tokens}
                log_metrics.update(metrics)
                # log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
