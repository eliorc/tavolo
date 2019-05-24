# Tavolo

Collection of deep learning modules and layers for the TensorFlow framework.
All the implementation are classes that inherit from `tf.keras.Layer` for ease-of-use writing high/low-level TensorFlow code.

## Installation

`pip install tavolo`

## Usage

`tavolo`'s layers are just like normal `tf.keras.layers`, simply import and use.

```python
import tavolo as tvl
import tensorflow as tf

.
.

model = tf.keras.Sequential([
    ..,
    ..,
    tvl.normalization.LayerNorm(),  # <-- Just like any tf.keras layer
    ..,
])

.
.

```

Notice, that these layers can also be used with low-level TensorFlow code

## Modules

### `embeddings`

Layers applied to embeddings

| Layer                          | Input -> Output dimensions                                                | Reference                        | 
| ------------------------------ | ------------------------------------------------------------------------- | -------------------------------- |
| `PositionalEncoding`           | (batch_size, time_steps, channels) -> (batch_size, time_steps, channels)  | https://arxiv.org/abs/1706.03762 |

### `normalization`

Normalization techniques

| Layer                          | Input -> Output dimensions                        | Reference                        | 
| ------------------------------ | ------------------------------------------------  | -------------------------------- |
| `LayerNorm`                    | (batch_size, channels) -> (batch_size, channels)  | https://arxiv.org/abs/1607.06450 |

### `seq2seq`

Layers mapping sequences to sequences

| Layer                          | Input -> Output dimensions                                                | Reference                        | 
| ------------------------------ | ------------------------------------------------------------------------- | -------------------------------- |
| `MultiHeadedSelfAttention`     | (batch_size, time_steps, channels) -> (batch_size, time_steps, channels)  | https://arxiv.org/abs/1706.03762 |

### `seq2vec`

Layers mapping sequences to vectors

| Layer                          | Input -> Output dimensions                                                | Reference                                                                       | 
| ------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `YangAttention`                | (batch_size, time_steps, channels) -> (batch_size, channels)              | https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf |


## TODO
- [ ] Write tests
- [ ] Contribution guide (shapes, tests)
- [ ] Write thank yous to code sources