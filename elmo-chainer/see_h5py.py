import h5py
import numpy as np
DTYPE = 'float32'

embedding_weight_file = 'elmo_token_embeddings.hdf5'
varname_in_file = 'embedding'

with h5py.File(embedding_weight_file, 'r') as fin:
    # Have added a special 0 index for padding not present
    # in the original model.

    embed_weights = fin[varname_in_file][...]
    weights = np.zeros(
        (embed_weights.shape[0] + 1, embed_weights.shape[1]),
        dtype=DTYPE)
    weights[1:, :] = embed_weights

print(weights.shape)
print(weights)


embedding_file = 'elmo_embeddings.hdf5'
sentence_idx = '1'  # 0-start index, so this is 2nd sentence

with h5py.File(embedding_file, 'r') as fin:

    embedding = fin[sentence_idx][...]

print(embedding.shape)
print(embedding)
