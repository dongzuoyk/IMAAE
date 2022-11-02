from torch.utils.data import Dataset
import numpy as np
import random
import math


class ScDataset(Dataset):
    def __init__(self, adata, orders=None):
        self.adata = adata

        dim = adata.shape[1]
        celltype_list = adata.obs.celltype.value_counts().index.tolist()
        batch_list = adata.obs.batch.unique().tolist()
        celltype_numbers = adata.obs.celltype.value_counts().tolist()

        attr2idx = {}
        idx2attr = {}
        for i, attr_name in enumerate(batch_list):
            attr2idx[attr_name] = i
            idx2attr[i] = attr_name
        
        
        adata_values = [np.array(adata.X[adata.obs['batch'] == batch]) for batch in batch_list]
        if orders is None:
            std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
            orders = np.argsort(std_)[::-1]
        else:
            orders = np.array([batches.index(item) for item in orders])
        
        orders = np.array([0,1,2,4,3])
        
        matrix_list = []
        trg_list = np.array([]).reshape(-1,2000)
        for celltype in celltype_list:
            
            c_batch = adata[np.array(adata.obs.celltype == celltype)].obs.batch.unique().tolist()
            
            sampling_num = celltype_numbers[celltype_list.index(celltype)] // len(c_batch)
            
            for i in range(sampling_num):
                tmp_matrix = []
                tmp_domain = []
                for batch in c_batch:
                    tmp_adata = adata[np.array(adata.obs.celltype == celltype) & np.array(adata.obs.batch == batch)]
                    tmp_matrix.append(random.choice(tmp_adata).X.squeeze())
                    tmp_domain.append(attr2idx[batch])
                
                _order = orders[orders < len(c_batch)]
                trg = np.array(tmp_matrix)[_order][0].reshape(1,-1)
                
                # trg = np.array(tmp_matrix).mean(axis=0).reshape(1,-1)
                trg_ = np.repeat(trg,len(c_batch),0)
                
                trg_list = np.append(trg_list,trg_,axis=0)
                matrix_list += tmp_matrix
        
        array_data = np.array(matrix_list)
        self.source = array_data.reshape(-1, dim)
          
        self.target = np.array(trg_list).reshape(-1, dim)
        print(self.target.shape)

        
        self.datasize = len(self.source)
        print(self.datasize)

    def __len__(self):
        return math.ceil(self.datasize / 1024) * 1024

    def __getitem__(self, index):

        ind = random.randint(0,self.datasize-1)
        return self.source[ind],self.target[ind]
