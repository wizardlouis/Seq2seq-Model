

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def pf(a1,a2):plt.figure(figsize=(a1,a2))
def tn(x): return x.cpu().detach().numpy()
def tt(x,dtype=torch.float,device="cpu"):
    return torch.tensor(x,dtype=dtype,device=device)

#saving different types of files
def save(object,filepath,filename,type,*args,**kwargs):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if type=='dict':
        np.save(filepath+'//'+filename,object)
    elif type=='dictxt':
        js=json.dumps(object)
        file=open(filepath+'//'+filename,'w')
        file.write(js)
        file.close()
    elif type=='npy':
        np.save(filepath+'//'+filename,object)
    pass

