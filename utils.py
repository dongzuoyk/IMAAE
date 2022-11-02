import scanpy as sc
import torch
import random
from annoy import AnnoyIndex
import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi



def data_preprocess(adata, n_top_genes=2000, key='batch', min_genes=600, min_cells=3,chunk_size=20000):
    adata = adata[:, [gene for gene in adata.var_names 
                  if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', batch_key=key)
    hv_genes = adata.var['highly_variable'].tolist()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    adata = adata[:, hv_genes]
    print('PreProcess Done.')

    return adata



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)




def compute_asw(arr, batch_label, cell_label):

    batch_score = silhouette_score(arr, batch_label)
    celltype_score = silhouette_score(arr, cell_label)


    return batch_score,celltype_score

def compute_ARI_NMI(adata):
    
    X = adata.obsm['X_pca']
    Y_batchcluster = adata.obs['batch'].tolist()
    Y_batchnum = len(set(Y_batchcluster))

    Y_cellcluster = adata.obs['celltype'].tolist()
    Y_cellnum = len(set(Y_cellcluster))

    km_batch = KMeans(n_clusters=Y_batchnum, init='k-means++', max_iter=30)
    km_batch.fit(X)
    y_batch = km_batch.predict(X)
    batch_ARI = adjusted_rand_score(Y_batchcluster,y_batch)
    batch_NMI = nmi(Y_batchcluster,y_batch)


    km_cell = KMeans(n_clusters=Y_cellnum, init='k-means++', max_iter=30)
    km_cell.fit(X)
    y_cell = km_cell.predict(X)
    cell_ARI = adjusted_rand_score(Y_cellcluster,y_cell)
    cell_NMI = nmi(Y_cellcluster, y_cell)


    return batch_ARI, cell_ARI,batch_NMI,cell_NMI


