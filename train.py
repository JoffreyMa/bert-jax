import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp
from bert import SimpleBERT
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import wandb

@jax.jit
def train_step(state, inputs, labels):
    def train_loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    loss, grads = jax.value_and_grad(train_loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, inputs, labels):
    logits = state.apply_fn({'params': state.params}, inputs)
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    return loss, predictions


def train(state, train_dataset, val_dataset, data_collator, batch_size, num_epochs, schedule, tokenizer, checkpoint_manager, save_args):
    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0
        with tqdm(total=round(len(train_dataset) / batch_size), desc='[Training]') as pbar:
            for batch in data_generator(train_dataset, data_collator, batch_size):
                # Train on batch
                batch_inputs, batch_labels = batch['input_ids'], batch['labels'] 
                state, loss = train_step(state, batch_inputs, batch_labels)
                lr = schedule(state.step)
                train_loss += loss.item()
                # Log
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                    "step": state.step
                })
                pbar.update(1)
                pbar.set_postfix({"loss":loss.item(), "lr":lr})
        avg_train_loss = train_loss / len(train_dataset)
        print(f"[Training] Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}, Learning rate: {lr}")
        wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss})

        val_loss = 0
        val_sequences_results = [] # reset at each epoch
        with tqdm(total=round(len(val_dataset) / batch_size), desc='[Validation]') as pbar:
            for batch in data_generator(val_dataset, data_collator, batch_size):
                batch_inputs, batch_labels = batch['input_ids'], batch['labels'] 
                loss, predictions = eval_step(state, batch_inputs, batch_labels)
                val_loss += loss.item()
                if len(val_sequences_results) < 1:
                    val_sequences_results.append({'input': tokenizer.decode(batch_inputs[0], skip_special_tokens=True), 'output': tokenizer.decode(predictions[0], skip_special_tokens=True)})
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_dataset)
        print(f"[Validation] Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")
        wandb.log({"epoch": epoch + 1, "avg_val_loss": avg_val_loss})
        for val_sequence in val_sequences_results:
            print(f"Input: {val_sequence['input']}\n######################################################################################################\nOutput: {val_sequence['output']}")

        # Save
        checkpoint_manager.save(state.step, state, save_kwargs={'save_args': save_args})

def collator_to_jax(data_collator_output):
    jax_data = {key: value.numpy() for key, value in data_collator_output.items()}
    return jax_data


def data_generator(dataset, data_collator, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]['input_ids']
        collated = data_collator(batch)  # Apply the data collator
        yield collator_to_jax(collated)  # Convert to JAX-compatible format


if __name__ == "__main__":
    batch_size=32 # 32 or 2 to test
    num_epochs=2 # 50 or 2 to test
    learning_rate=1e-4
    dataset_path = "/home/infres/jma-21/bert-jax/data"
    test_size = 0.1 # 0.1 or 0.999 o test
    ckpt_path = "/home/infres/jma-21/bert-jax/checkpoint"
    ckpt_step = None # examples: "384" or None
    
    model = None
    vocab_size = 32000
    max_seq_length = 512
    dim = 256 # 128 or 8 to test
    num_heads = 16 # 12 or 4 to test
    num_layers = 12 # 4 or 2 to test
    hidden_dim = 1024 # 512 or 64 to test

    # Load model
    model = SimpleBERT(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_dataset = dataset.train_test_split(test_size=test_size)
    test_dataset = split_dataset['test']
    trainval_dataset = split_dataset['train'].train_test_split(test_size=0.1)
    train_dataset, val_dataset = trainval_dataset['train'], trainval_dataset['test'] 
    # Adapt dataset to MLM
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.05
    )

    # Checkpointer
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    # Restore checkpoint
    key = jax.random.PRNGKey(0)
    input_ids = np.random.randint(0, vocab_size, (1, max_seq_length))

    # Initialize model and optimizer
    params = model.init(key, input_ids)['params']
    transition_steps = num_epochs * (len(train_dataset) // batch_size)
    pct_start = 0.1
    pct_final = 0.9
    div_factor = 100
    final_div_factor = 10000
    schedule = optax.linear_onecycle_schedule(transition_steps=transition_steps, peak_value=learning_rate, pct_start=pct_start, pct_final=pct_final, div_factor=div_factor, final_div_factor=final_div_factor)
    tx = optax.adam(schedule)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    if ckpt_step:
        # Placeholder for state structure to initialize the restored state
        state = orbax_checkpointer.restore(ckpt_path + '/' + ckpt_step + '/default', item=state)
        

    # Prepare state versioning and automatic bookkeeping
    save_args = orbax_utils.save_args_from_target(state)
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = ocp.CheckpointManager(ckpt_path, orbax_checkpointer, options)

    # Initialize W&B
    wandb.init(
        project="bert-flax-training",
        config={
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'schedule': 'linear_onecycle_schedule',
            'transition_steps' : transition_steps,
            'pct_start': pct_start,
            'pct_final': pct_final,
            'div_factor': div_factor,
            'final_div_factor': final_div_factor,
            'vocab_size': vocab_size,
            'max_seq_length': max_seq_length,
            'dim': dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'train_dataset_len': len(train_dataset),
            'val_dataset_len': len(val_dataset),
            'test_dataset_len': len(test_dataset),
        }
    )

    train(state, train_dataset, val_dataset, data_collator, batch_size, num_epochs=num_epochs, schedule=schedule, tokenizer=tokenizer, checkpoint_manager=checkpoint_manager, save_args=save_args)
