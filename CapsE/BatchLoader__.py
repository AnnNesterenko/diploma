import tensorflow._api.v2.compat.v1 as tf
import numpy as np

BATCH_SIZE=128


class BatchLoaderEcir(object):
    def __init__(self, train_triples, train_val_triples, batch_size=BATCH_SIZE):

        self.train_triples = train_triples
        self.train_val_triples = train_val_triples
        self.batch_size = batch_size

    def __call__(self):

        idxs = np.random.randint(0, len(self.train_val_triples), self.batch_size)
        self.new_triples_indexes = np.concatenate(self.train_triples[idxs])
        self.new_triples_values = np.concatenate(self.train_val_triples[idxs], axis=0)

        while len(self.new_triples_indexes) < self.batch_size * 10:
            self.new_triples_indexes = np.append(self.new_triples_indexes, self.new_triples_indexes, axis=0)
            self.new_triples_values = np.append(self.new_triples_values, self.new_triples_values, axis=0)

        self.new_triples_indexes = np.append(self.new_triples_indexes, self.new_triples_indexes[:(self.batch_size * 20 - self.new_triples_values.shape[0])], axis=0)
        self.new_triples_values = np.append(self.new_triples_values, self.new_triples_values[:(self.batch_size * 20 - self.new_triples_values.shape[0])], axis=0)

        return self.new_triples_indexes.astype(np.int32), self.new_triples_values.astype(np.float32)