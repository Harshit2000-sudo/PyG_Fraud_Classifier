# %%
"""
## Download dataset
"""

# %%
!wget https://raw.githubusercontent.com/geniusai-research/interns_task/main/sampled_data2.csv

# %%
"""
## Download libraries
"""

# %%
!pip install sentence-transformers
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install torch-geometric

# %%
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, DataLoader

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ReLU
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, HeteroConv, SAGPooling

# %%
data = pd.read_csv("sampled_data2.csv")
data.head(10)

# %%
df = data.copy()
X = df.drop(columns=["step","age","gender", "category", "amount", "fraud"])
for f in X.columns:
    X[f] =  X[f].fillna('') 
    encoder =  LabelEncoder()
    encoder.fit(X[f])
    X[f] = encoder.transform(X[f])

# %%
df["customer_id"] = X["customer"]
df["merchant_id"] = X["merchant"]
df.head()

# %%
"""
# .csv* to PyG data format
## #Firstly we need to convert this data into graph format
"""

# %%
# We'll first extract the nodes and there mapping.
# for that we're defing load_node_csv function

def load_node_csv(data, index_col,encoders=None):
    
    df = data.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1).to(torch.float32)


    return x, mapping

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device,)
        
        return x.cpu().to(torch.float32)


class Cat_Encoder(object):
    def __init__(self, sep=None):
        self.sep = sep

    def __call__(self, df):
      if self.sep:
        categ = set(g for col in df.values for g in col.split(self.sep))        
        
        mapping = {cat: i for i, cat in enumerate(categ)}
        
        x = torch.zeros(len(df), len(mapping), dtype= torch.float32)
        
        for i, col in enumerate(df.values):
          for cat in col.split(self.sep):
            x[i, mapping[cat]] = 1
      
      elif self.sep == None:
        categ = set(i for i in df.values)        
        
        mapping = {cat: i for i, cat in enumerate(categ)}
        
        x = torch.zeros(len(df), len(mapping))
        
        for i, col in enumerate(df.values):
            x[i, mapping[col]] = 1
      
      return x.to(torch.float32)



# %%
merchant_x, merchant_mapping = load_node_csv(
    df, index_col='merchant_id', encoders={
        "category": Cat_Encoder(sep = "&"),
        "amount" : Cat_Encoder()
    })

customer_x, customer_mapping = load_node_csv(
     df, index_col='customer_id', encoders={
        'age': Cat_Encoder(sep = "to"),
        "gender": SequenceEncoder()
    })

# %%
def load_edge_csv(data, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    
    df = data.copy()

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index.to(torch.int64), edge_attr.to(torch.float32)

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

# %%
g_data = HeteroData()

g_data["customer"].x = customer_x
g_data["merchant"].x = merchant_x


# %%
edge_index, edge_label = load_edge_csv(
    df,
    src_index_col= 'customer_id',
    src_mapping= customer_mapping,
    dst_index_col='merchant_id',
    dst_mapping= merchant_mapping,
    encoders={'amount': IdentityEncoder(dtype=torch.float32)},
)

g_data['customer', 'amount', 'merchant'].edge_index = edge_index
g_data['merchant', 'amount', 'customer'].edge_index = edge_index

y = torch.from_numpy(df["fraud"].values).to(torch.long)
g_data["customer"].y = y

print(g_data)

# %%
node_types, edge_types = g_data.metadata()
print(edge_types)

# %%
"""
# DataLoader
"""

# %%
import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g_data = T.NormalizeFeatures()(g_data)
self_loop = T.AddSelfLoops()
g_data = self_loop(g_data)

transforms = T.RandomNodeSplit()
g_data = transforms(g_data)

# %%
g_data

# %%
class HeteroSaG(torch.nn.Module):
  def __init__(self, metadata, hidden_channels, out_channels, num_layers):
      super().__init__()

      self.convs = torch.nn.ModuleList()
      for _ in range(num_layers):
          conv = HeteroConv({
              edge_type: SAGEConv((-1, -1), hidden_channels)
              for edge_type in metadata[1]
          })
          self.convs.append(conv)
      
      # pooling layer
      #self.pool = SAGPooling(hidden_channels, ratio=.5, )
      self.lin_1 = Linear(hidden_channels, hidden_channels//2)
      self.relu_1 = nn.ReLU()
      self.lin_2 = Linear(hidden_channels//2, hidden_channels//4)
      self.relu_2 = nn.ReLU()
      self.lin_3 = Linear(hidden_channels//4, out_channels)


  def forward(self, x_dict, edge_index_dict):
      for conv in self.convs:
          x_dict = conv(x_dict, edge_index_dict)
          x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
      
      x_1 = self.lin_1(x_dict["customer"])
      x_1 = self.relu_1(x_1)
      
      x_2 = self.lin_2(x_1)
      x_2 = self.relu_2(x_2)
      x_3 = self.lin_3(x_2)

      return x_3

model = HeteroSaG(g_data.metadata(), hidden_channels=50, out_channels=2, num_layers=4)

g_data, model = g_data.to(device), model.to(device)
print(model)

# %%
g_data.edge_index_dict

# %%
with torch.no_grad():  # Initialize lazy modules.
   out = model(g_data.x_dict, g_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(g_data.x_dict, g_data.edge_index_dict)
    mask = g_data["customer"].train_mask
    loss = F.cross_entropy(out[mask], g_data["customer"].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss), out


@torch.no_grad()
def test():
    model.eval()
    pred = model(g_data.x_dict, g_data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = g_data["customer"][split]
        acc = (pred[mask] == g_data["customer"].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

l = []
tr_ac = []
te_ac = []
va_ac = []

for epoch in range(1, 5000):    
    loss, out= train()
    l.append(loss)
    
    train_acc, val_acc, test_acc = test()
    tr_ac.append(train_acc)
    va_ac.append(val_acc)
    te_ac.append(test_acc)
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# %%
import matplotlib.pyplot as plt
x = [id for id, i in enumerate(l)]
plt.plot(x, l)
plt.show()

# %%
x_acc = [id for id, i in enumerate(tr_ac)]
plt.plot(x, tr_ac)
plt.show()