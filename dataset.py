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

        batch2idx = {}
        idx2batch = {}
        for i, attr_name in enumerate(batch_list):
            batch2idx[attr_name] = i
            idx2batch[i] = attr_name
        
        
        adata_values = [np.array(adata.X[adata.obs['batch'] == batch]) for batch in batch_list]
        if orders is None:
            std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
            orders = np.argsort(std_)[::-1]
            print(orders)
        else:
            orders = np.array([batches.index(item) for item in orders])
            
        orders = np.array([0,1])
        
        mode = 'Mean'
        input_index = []
        anchor = []
        index_adata = np.array(adata.obs.index.tolist())
        for celltype in celltype_list:
            
            # 每种细胞类型出现的批次
            c_batch = adata[np.array(adata.obs.celltype == celltype)].obs.batch.unique().tolist()
            
#             if(len(c_batch) == 1):
#                 continue
            
            # 每种细胞类型进行均衡采样的数量
            sampling_num = celltype_numbers[celltype_list.index(celltype)] // len(c_batch)
            
            for i in range(sampling_num):
                
                tmp_input_index = []
                
                for batch in c_batch:
                    # 每个批次的 选定 细胞类型的数据 索引号
                    bc_index_adata = index_adata[np.array(adata.obs.celltype == celltype) & np.array(adata.obs.batch == batch)]
                    tmp_input_index.append(random.choice(bc_index_adata))
                
                # 去除 当前细胞类型不存在的域，并保证原来顺序
                _order = orders[orders < len(c_batch)]
                if mode == 'Mean':
                    anchor_point = np.array(adata[tmp_input_index].X).mean(axis=0).reshape(1,-1)
                    tmp_anchor = np.repeat(anchor_point,len(c_batch),0)
                    
                    anchor += tmp_anchor.tolist()
                else:
                    anchor_point = np.array(tmp_input_index)[_order][0]                  
                    tmp_anchor = np.repeat(anchor_point,len(c_batch),0)
                    anchor += tmp_anchor.tolist()
                
                input_index += tmp_input_index
        
        self.source = np.array(adata[input_index].X).reshape(-1, dim)
        
        if mode == 'Mean':
            self.target = np.array(anchor).reshape(-1, dim)
        else:
            self.target = np.array(adata[np.array(anchor).reshape(1, -1).squeeze()].X)
        print(self.target.shape)
        
        self.datasize = len(self.source)
        print(self.datasize)

    def __len__(self):
        return math.ceil(self.datasize / 1024) * 1024

    def __getitem__(self, index):

        ind = random.randint(0,self.datasize-1)
        return self.source[ind],self.target[ind]
