import jax
import optax
import numpy as np
from flax.training import train_state
from bert import SimpleBERT
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm


#@jax.jit
def apply_model(state, inputs, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads


#@jax.jit
def train_step(state, inputs, labels):
    loss, grads = apply_model(state, inputs, labels)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Example training loop
def train(model, dataset, data_collator, batch_size, num_epochs, learning_rate):
    key = jax.random.PRNGKey(0)
    input_ids = np.random.randint(0, 32000, (1, 512))

    # Initialize model and optimizer
    params = model.init(key, input_ids)['params']
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training loop
    for epoch in range(num_epochs):
        with tqdm(total=round(len(dataset) / batch_size), desc='Training') as pbar:
            for batch in data_generator(dataset, data_collator, batch_size):
                batch_inputs, batch_labels = batch['input_ids'], batch['labels'] 
                state, loss = train_step(state, batch_inputs, batch_labels)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}, Loss: {loss}")


def collator_to_jax(data_collator_output):
    jax_data = {key: value.numpy() for key, value in data_collator_output.items()}
    return jax_data


def data_generator(dataset, data_collator, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]['input_ids']
        collated = data_collator(batch)  # Apply the data collator
        yield collator_to_jax(collated)  # Convert to JAX-compatible format


if __name__ == "__main__":
    batch_size=10
    
    # Load model
    model = SimpleBERT(
        vocab_size=32000,
        max_seq_length=512,
        dim=8,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
    )

    # Load dataset
    dataset = load_from_disk("/home/infres/jma-21/bert-jax/data")
    split_dataset = dataset.train_test_split(test_size=0.999)
    # Adapt dataset to MLM
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    train(model, split_dataset['train'], data_collator, batch_size, num_epochs=3, learning_rate=1e-4)
