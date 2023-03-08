import networkx as nx
import numpy as np
from torch.utils.data import Dataset
import random
from itertools import combinations
from collections import Counter
import time

from utils import acquire_pairs

import multiprocessing
from anndata import AnnData
import math

def create_pairs_dict(pairs, pairs_dict):
    for x, y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict


class ScDataset(Dataset):
    def __init__(self, ppd_adata, mnn_times=1, len_weight=5, self_nbs=5, other_nbs=1, overlap=False, batch_num=2, under_sample=False, under_sample_num=20000):

        self.adata = ppd_adata
        self.ppd_adata = ppd_adata
        self.pca_adata = AnnData(ppd_adata.obsm['X_pca'], ppd_adata.obs)

        self.same_batch_pairs = {}
        self.diff_batch_pairs = {}

        self.batch_list = self.pca_adata.obs['batch'].unique().tolist()
        self.celltype_list = self.pca_adata.obs.celltype.value_counts().index.tolist()

        self.source = np.empty(shape=(0,2000))
        self.target = np.empty(shape=(0,2000))

        for celltype in self.celltype_list:
            print(celltype)
            self.pca_adata_celltype = self.pca_adata[np.array(self.pca_adata.obs.celltype == celltype)]
            self.ppd_adata_celltype = self.ppd_adata[np.array(self.ppd_adata.obs.celltype == celltype)]
            self.index_list = self.pca_adata_celltype.obs.index.tolist()
            self.index_dir = {}

            for i in range(len(self.index_list)):
                self.index_dir[self.index_list[i]] = i
            self.sample_num = int(len(self.pca_adata_celltype) / len(self.pca_adata_celltype.obs['batch'].unique()))

            if self.sample_num > 3000:
                self.sample_num = 3000
            print("Number of samples per batch: "+str(self.sample_num))

            start_time = time.time()

            # Initialization
            all_list = [i for i in range(len(self.pca_adata_celltype))]
            exc_list = {}

            same_batch_tag = {}

            for batch1, batch2 in combinations(self.batch_list, 2):
                self.same_batch_pairs[batch1] = []
                self.same_batch_pairs[batch2] = []
                self.diff_batch_pairs[str(batch1)+"_"+str(batch2)] = []
                exc_list[str(batch1)+"_"+str(batch2)] = []
                same_batch_tag[batch1] = True
                same_batch_tag[batch2] = True

            # Find nearest neighbors
            for i in range(mnn_times):
                print("Times: " + str(i+1))
                tmp_sample = {}
                for batch1, batch2 in combinations(self.batch_list, 2):
                    print(batch1 + "<——>" + batch2)

                    # Remove paired cells in two batches
                    new_list = list(
                        set(exc_list[str(batch1)+"_"+str(batch2)]) ^ set(all_list))

                    batch_1 = self.pca_adata_celltype[new_list,:][self.pca_adata_celltype[new_list, :].obs['batch'] == batch1]
                    batch_2 = self.pca_adata_celltype[new_list,:][self.pca_adata_celltype[new_list, :].obs['batch'] == batch2]

                    if batch1 not in tmp_sample.keys():
                        tmp_1 = np.arange(len(batch_1))
                        np.random.shuffle(tmp_1)
                        tmp_sample[batch1] = tmp_1
                        if same_batch_tag[batch1]:
                            self.same_batch_pairs[batch1] += acquire_pairs(batch_1[tmp_1[:self.sample_num]], batch_1[tmp_1[:self.sample_num]], self_nbs, 'angular', self.index_dir)
                            if len(batch_1) <= 3000:
                                same_batch_tag[batch1] = False

                    if batch2 not in tmp_sample.keys():
                        tmp_2 = np.arange(len(batch_2))
                        np.random.shuffle(tmp_2)
                        tmp_sample[batch2] = tmp_2
                        if same_batch_tag[batch2]:
                            self.same_batch_pairs[batch2] += acquire_pairs(batch_2[tmp_2[:self.sample_num]], batch_2[tmp_2[:self.sample_num]], self_nbs, 'angular', self.index_dir)
                            if len(batch_2) <= 3000:
                                same_batch_tag[batch2] = False

                    index_1 = tmp_sample[batch1]
                    index_1 = index_1[index_1 < len(batch_1)]
                    index_2 = tmp_sample[batch2]
                    index_2 = index_2[index_2 < len(batch_2)]

                    self.diff_batch_pairs[str(batch1)+"_"+str(batch2)] += acquire_pairs(batch_1[index_1[:self.sample_num]], batch_2[index_2[:self.sample_num]], other_nbs, 'angular', self.index_dir)

                for key in self.diff_batch_pairs.keys():
                    for k, v in self.diff_batch_pairs[key]:
                        exc_list[key] += [k, v]

            end_time = time.time()
            print("Find MNNs time-consuming :" + str(end_time-start_time))

            # Construct connected graphs of similar cells across batches
            self.nodes = []
            self.edges = []
            for pair_name in self.diff_batch_pairs.keys():
                for x, y in self.diff_batch_pairs[pair_name]:
                    self.nodes.append(x)
                    self.nodes.append(y)
                    self.edges.append((x, y, 1))
            self.nodes = list(set(self.nodes))

            print("The percentage of MNN paired cells in the total cells : " +
                str(len(self.nodes)/len(self.pca_adata_celltype)))
            self.G = nx.Graph()
            self.G.add_nodes_from(self.nodes)
            self.G.add_weighted_edges_from(self.edges)
            self.C = sorted(nx.connected_components(self.G), key=len, reverse=True)

            self.pairs_dict = {}  # key:cell id, value: its neighbors id in the same batch
            for key in self.same_batch_pairs.keys():
                self.pairs_dict = create_pairs_dict(
                    self.same_batch_pairs[key], self.pairs_dict)

            for i in range(len(self.pca_adata_celltype)):
                if i not in self.pairs_dict.keys():
                    self.pairs_dict[i] = [i]

            self.init_dataset(len_weight, overlap, batch_num, under_sample, under_sample_num)

    def init_dataset(self, len_weight, overlap, batch_num, under_sample, under_sample_num):

        # Sampled subgraph nodes list，each subgraph contains cells from different batches
        graph_list = []
        for graph in self.C:

            tmp_dict = {}  # key：batch name，value：cell id list
            for node in graph:
                try:
                    tmp_dict[self.pca_adata_celltype[node].obs['batch'].tolist()[0]] += [node]
                except:
                    tmp_dict[self.pca_adata_celltype[node].obs['batch'].tolist()[0]] = []
                    tmp_dict[self.pca_adata_celltype[node].obs['batch'].tolist()[0]] += [node]

            # Batches contained in the current connected graph
            in_batch = list(tmp_dict.keys())

            # Batches not contained in the current connected graph
            out_batch = list(set(self.batch_list) ^ set(in_batch))

            # List of cell's id sampled in the current connected graph
            tmp_graph_list = []
            if out_batch == []:
                sample_times = int(np.mean([len(cell_id_list) for cell_id_list in tmp_dict.values()]))
                for i in range(sample_times):
                    tmp_graph_list.append([random.choice(cell_id_list) for cell_id_list in tmp_dict.values()])
            else:
                new_tmp_dict = tmp_dict.copy()
                # Find the connected graph where the k-nearest neighbor of the cell in the current connected graph is located
                # Observe whether the connected graph contains cells that do not contain batches in the current connected graph
                for batch_name in tmp_dict.keys():
                    for cell_id in tmp_dict[batch_name]:
                        neighbors = self.pairs_dict[cell_id]
                        if len(neighbors) == 1:
                            continue
                        for nb in neighbors:
                            try:
                                nb_graph_nodes = list(self.G.adj[nb])
                                for nb_node in nb_graph_nodes:
                                    if self.pca_adata_celltype[nb_node].obs['batch'].tolist()[0] in out_batch:
                                        try:
                                            new_tmp_dict[self.pca_adata_celltype[nb_node].obs['batch'].tolist()[0]] += [nb_node]
                                        except:
                                            new_tmp_dict[self.pca_adata_celltype[nb_node].obs['batch'].tolist()[0]] = []
                                            new_tmp_dict[self.pca_adata_celltype[nb_node].obs['batch'].tolist()[0]] += [nb_node]
                            except:
                                continue
                sample_times = int(np.mean([len(cell_id_list) for cell_id_list in new_tmp_dict.values()]))
                for i in range(sample_times):
                    tmp_graph_list.append([random.choice(cell_id_list) for cell_id_list in new_tmp_dict.values()])

            graph_list += tmp_graph_list

        print("The final number of subgraphs :" + str(len(graph_list)))
        len_list = [len(g) for g in graph_list]
        print(Counter(len_list))

        self.source_id = []
        self.target_id = []

        for graph in graph_list:

            self.source_graph = []
            self.target_graph = []

            for node in graph:
                for i in range(len(graph) * len_weight):
                    self.source_graph.append(random.choice(self.pairs_dict[node]))

            self.target_graph = [[] for i in range(len(self.source_graph))]

            for i in range(len(self.source_graph)):
                for node in graph:
                    self.target_graph[i] += [random.choice(self.pairs_dict[node])]

            self.source_id += self.source_graph
            self.target_id += self.target_graph

        if under_sample:
            random.seed(10)
            self.source_id = random.sample(self.source_id, under_sample_num)
            random.seed(10)
            self.target_id = random.sample(self.target_id, under_sample_num)

        
        if len(self.source) == 0 :
            self.source = self.ppd_adata_celltype[self.source_id].X.toarray().squeeze()
        else:
            self.source = np.concatenate((self.ppd_adata_celltype[self.source_id].X.toarray().squeeze(),self.source),axis=0)


        t3 = time.time()

        if len(self.target) == 0:
            self.target = np.array([self.ppd_adata_celltype[l].X.mean(axis=0).toarray().squeeze() for l in self.target_id])
        else:
            self.target = np.concatenate((np.array([self.ppd_adata_celltype[l].X.mean(axis=0).toarray().squeeze() for l in self.target_id]),self.target),axis=0)

        t4 = time.time()

        self.datasize = len(self.source)
        print("Dataset size : " + str(self.datasize))
        print("Dataset size shape: " + str(self.source.shape))
        print("Time-consuming to build dataset: " + str(t4-t3))

    def __len__(self):
        return math.ceil(self.datasize / 1024) * 1024

    def __getitem__(self, index):

        ind = random.randint(0, self.datasize-1)
        return self.source[ind], self.target[ind]
