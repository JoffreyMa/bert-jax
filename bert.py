import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    num_heads: int
    dim: int

    def setup(self):
        assert self.dim % self.num_heads == 0, "Dimension must be divisible by number of heads"
        self.head_dim = self.dim // self.num_heads
        self.query = nn.Dense(self.dim)
        self.key = nn.Dense(self.dim)
        self.value = nn.Dense(self.dim)
        self.out = nn.Dense(self.dim)

    def __call__(self, x):
        batch_size, seq_len, dim = x.shape

        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        attention_scores = jnp.einsum('bhqd, bhkd -> bhqk', q, k) / jnp.sqrt(self.head_dim)
        attention_weights = nn.softmax(attention_scores, axis=-1)

        context = jnp.einsum('bhqk, bhvd -> bhqd', attention_weights, v)
        context = context.reshape(batch_size, seq_len, self.dim)

        return self.out(context)


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)
        self.dense2 = nn.Dense(self.dim)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        return self.dense2(x)


class EncoderLayer(nn.Module):
    dim: int
    num_heads: int
    hidden_dim: int

    def setup(self):
        self.self_attention = MultiHeadSelfAttention(self.num_heads, self.dim)
        self.feed_forward = FeedForward(self.dim, self.hidden_dim)
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()

    def __call__(self, x):
        # Multi-head self-attention with residual connection
        attn_output = self.self_attention(x)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward layer with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        return x


# Simple BERT with MLM head
class SimpleBERT(nn.Module):
    vocab_size: int
    max_seq_length: int
    dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.dim)
        self.position_embedding = nn.Embed(self.max_seq_length, self.dim)
        self.encoder_layers = [
            EncoderLayer(self.dim, self.num_heads, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        self.mlm_head = nn.Dense(self.vocab_size)

    def __call__(self, input_ids):
        seq_length = input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)
        position_ids = jnp.arange(seq_length)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings

        for layer in self.encoder_layers:
            x = layer(x)

        logits = self.mlm_head(x)
        return logits


# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = SimpleBERT(
        vocab_size=32000,
        max_seq_length=512,
        dim=8,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
    )

    input_ids = np.random.randint(0, 32000, (1, 128))
    variables = model.init(key, input_ids)
    output = model.apply(variables, input_ids)

    print(output.shape)  # Should be (1, 128, 768)
