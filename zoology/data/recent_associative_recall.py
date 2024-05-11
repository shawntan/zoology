import numpy as np
import torch
from .utils import SyntheticData, builder_from_single

def multiquery_ar(
    vocab_size: int=8_192,
    num_train_examples: int=100_000,
    num_test_examples: int=3_000,
    input_seq_len: int=64,
    num_kv_pairs: int=4,
    train_power_a: float=0.01,
    test_power_a: float=0.01,
    random_non_queries: bool=True,
    seed: int=0,
) -> SyntheticData:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the 
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """

    train_inputs, train_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_train_examples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=train_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )
    test_inputs, test_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_test_examples,
        input_seq_len=input_seq_len,
        seed=seed + 10,  # different seed for test set
        power_a=test_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )

    data = SyntheticData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
    )

    # check for data leakage:
    train_set = set([" ".join(map(str, x)) for x in data.train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in data.test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )
    return data


def _mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True
):
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=True, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=True, size=num_kv_pairs)
    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(
        queries, (gaps * 2),
        values=np.apply_along_axis(np.random.choice, 1, keys, replace=True, size=keys[0].shape),
        axis=1
    )
    examples = np.concatenate([kvs, queries], axis=1)
    inputs = torch.tensor(examples[:, :-1])

    if random_non_queries:
        inputs[inputs == 0] = torch.tensor(
            np.random.choice(value_choices, replace=True, size=inputs[inputs == 0].shape))

    def process_sequence(seq):
        out = np.full((seq.shape[0],), -100, dtype=np.int64)
        state = {}
        curr_key = None
        for i in range(seq.shape[0]):
            if seq[i] in key_choices:
                curr_key = seq[i]
                if curr_key in state:
                    out[i] = state[curr_key]
            elif curr_key is not None:
                state[curr_key] = seq[i]
                curr_key = None
        return out
    labels = torch.tensor(
        np.apply_along_axis(process_sequence, axis=1, arr=inputs))
    # labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    # labels = np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)
    # inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    # replace all the 0 with random values
    return inputs, labels


    
@builder_from_single
def base_ar(
    vocab_size: int,
    input_seq_len: int,
    rng: np.random.Generator,
):
    """Generate sequence where the input has a sequence of key value pairs
    and the copy prefix at the end, and then a key value pair is inserted
    after the copy prefix."""
    non_special_vocab_size = vocab_size - 1
    keys = np.arange(non_special_vocab_size // 2)
    values = np.arange(non_special_vocab_size // 2, non_special_vocab_size)
    keys = [ [key] for key in keys ]
    kv_map = {tuple(k): rng.choice(values) for k in keys}

    key_present = {}
    vocab_seq = []
    pair_length = 2
    input_seq_len -= 2
    for _ in range(input_seq_len // (pair_length)):
        k = tuple(rng.choice(list(kv_map.keys())))
        v = kv_map[k]
        vocab_seq += list(k) + [v]
        key_present[k] = True

    k = tuple(rng.choice(list(kv_map.keys())))
    while k not in key_present:
        k = tuple(rng.choice(list(key_present.keys())))
    to_copy = [vocab_size - 1] + list(k) + [ kv_map[k]  ]
    vocab_seq = vocab_seq + to_copy
    seq = torch.tensor(vocab_seq)
    return seq[:-1], seq[1:]

if __name__ == "__main__":
    mqar_data = multiquery_ar(
        vocab_size=1024,
        num_train_examples=1,
        num_test_examples=1,
        input_seq_len=16,
        num_kv_pairs=4,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=True,
        seed=0,
    )
    print("Input:", mqar_data.train_inputs)
    print("Label: ", mqar_data.train_labels)
