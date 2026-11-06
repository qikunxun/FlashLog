import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class KGDataset(Dataset):

    def __init__(self, option=None):
        self.option = option
        self.batch_size = self.option.batch_size
        self.kg, self.valid_data, self.test_data, self.entity_dict, self.id2relation, self.relation2id, self.train_kg, \
                self.relation_sorted = self.load_kg_all(self.option.data_dir)
        self.relation_size = (len(self.relation2id) - 1) // 2
        self.valid_size = self.valid_data.shape[0]
        self.test_size = self.test_data.shape[0]

        if self.option.early_stop or self.option.do_test:
            if getattr(self.option, 'use_csr_eval', False):
                self.graph_entity = None
                self.graph = None
            else:
                kg_all = pd.concat([self.valid_data, self.test_data])
                self.graph_entity = (kg_all.groupby([0, 1]), kg_all.groupby([2, 1]))

                self.graph = (self.kg.groupby([0, 1]), self.kg.groupby([2, 1]))

        # The training loop reads triples; it never mutates this array. Reuse
        # pandas' homogeneous int32 block instead of duplicating the full graph.
        train_data = self.kg.to_numpy(copy=False)
        if option.wot_inv:
            train_data_inv = np.concatenate([train_data[:, 2:], train_data[:, 1:2] + self.relation_size,
                                             train_data[:, 0:1]], axis=-1)
            self.train_data = np.concatenate([train_data, train_data_inv], axis=0)
        else:
            self.train_data = train_data

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.train_data[idx]

    def load_kg_all(self, kg_path):
        train_path = os.path.join(kg_path, 'train_id.txt')
        valid_path = os.path.join(kg_path, 'valid_id.txt')
        test_path = os.path.join(kg_path, 'test_id.txt')
        entity_path = os.path.join(kg_path, 'entities.dict')
        relation_path = os.path.join(kg_path, 'relations.dict')
        start_time = time.time()
        # Model execution uses numeric IDs and the entity count, not names.
        # Keep the original row order and IDs without millions of Python strings.
        entity_dict = pd.read_csv(entity_path, encoding='utf-8', engine='c', sep='\t', header=None,
                                  usecols=[0], dtype={0: np.int32})
        print('Loading entity dict done. Data size: {}, time cost: {}'.format(entity_dict.shape[0], time.time() - start_time))
        id2relation, relation2id = {}, {}
        with open(relation_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                idx, relation = line.strip().split('\t')
                relation2id[relation] = int(idx)
                id2relation[int(idx)] = relation
        length = len(relation2id)
        for i in range(length):
            relation2id['INV' + id2relation[i]] = len(relation2id)
            id2relation[len(relation2id) - 1] = 'INV' + id2relation[i]
        relation2id['Identity'] = len(relation2id)
        id2relation[len(id2relation)] = 'Identity'
        start_time = time.time()
        train_data = pd.read_csv(train_path, encoding='utf-8', engine='c', sep='\t', header=None,
                                dtype={0: np.int32, 1: np.int32, 2: np.int32})
        # CSV parsing can leave one pandas block per column. Construct one
        # homogeneous block so subsequent NumPy views really share storage.
        train_data = pd.DataFrame(train_data.to_numpy(copy=False), copy=False)
        data_size = train_data.shape[0]
        print('Loading training set done. Data size: {}, time cost: {}'.format(data_size, time.time() - start_time))
        start_time = time.time()
        train_triples = train_data.groupby(1)
        print('Grouping training set by relation, time cost: {}'.format(time.time() - start_time))
        # Do not materialize every per-relation DataFrame just to count rows.
        # Match groupby's ascending relation order for equal-size groups.
        counts = np.zeros(length, dtype=np.int64)
        relation_ids = train_data[1].to_numpy(copy=False)
        for start in range(0, data_size, 1 << 20):
            counts += np.bincount(relation_ids[start:start + (1 << 20)],
                                  minlength=length)
        relations = np.flatnonzero(counts)
        relation_sorted = [(int(r), int(counts[r]) / data_size)
                           for r in sorted(relations, key=lambda r: -int(counts[r]))]

        valid_data = pd.read_csv(valid_path, encoding='utf-8', engine='c', sep='\t', header=None,
                                dtype={0: np.int32, 1: np.int32, 2: np.int32})
        test_data = pd.read_csv(test_path, encoding='utf-8', engine='c', sep='\t', header=None,
                                dtype={0: np.int32, 1: np.int32, 2: np.int32})
        return train_data, valid_data, test_data, entity_dict, id2relation, relation2id, \
               train_triples, relation_sorted

    def batch_iter(self):
        sample_size = self.option.iteration_per_batch * self.batch_size
        if len(self.kg.index.values) < sample_size:
            batches_all = self.kg.index.values.copy()
            np.random.shuffle(batches_all)
        else:
            batches_all = np.random.choice(self.kg.index.values, sample_size, replace=False)

        for start in range(0, len(batches_all), self.batch_size):
            batches = batches_all[start: start + self.batch_size]
            triples = []
            for index in batches:
                h = self.kg[0][index]
                r = self.kg[1][index]
                t = self.kg[2][index]
                if self.option.wot_inv:
                    flag = np.random.randint(0, 2, 1)[0]
                    if flag == 0:
                        triples.append((h, r, t))
                    else:
                        triples.append((t, r + self.relation_size, h))
                else:
                    triples.append((h, r, t))
                    triples.append((t, r + self.relation_size, h))
            yield batches, triples, False

    def build_graph(self, graph_entity, data):
        for index_r, grouped_data in data:
            for index in grouped_data.index:
                h, r, t = grouped_data[0][index], grouped_data[1][index], grouped_data[2][index]
                if h not in graph_entity:
                    graph_entity[h] = {t: [r]}
                else:
                    if t in graph_entity[h]:
                        graph_entity[h][t].append(r)
                    else:
                        graph_entity[h][t] = [r]

                r_inv = r + self.relation_size
                if t not in graph_entity:
                    graph_entity[t] = {h: [r_inv]}
                else:
                    if h in graph_entity[t]:
                        graph_entity[t][h].append(r_inv)
                    else:
                        graph_entity[t][h] = [r_inv]
        return graph_entity
