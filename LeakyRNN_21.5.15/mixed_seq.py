from gene_seq import *
import argparse
import numpy as np
from rw import *
import os

parser=argparse.ArgumentParser()
parser.add_argument("--out_dir",help="seq data out put filepath",type=str,default='')
parser.add_argument("--load_dir",help="seq data loading filepath",type=str,default='')
parser.add_argument("--n_item",help="number of item",type=int,default=6)
#We may split generating process to details
parser.add_argument("--mixed_length",help="upper limit of sequence length",type=int,nargs='+')#defaullt=[1,2,3]
parser.add_argument("--trainset_ratio",help="ratio of training set",type=float,nargs='+')#default=[1,0.6,0.4]
parser.add_argument("--repeat",help="if item repeated is available in sequence",type=int,default=1)

args=parser.parse_args()

#hyperparameter that determine generating Seq_set
out_dir='SD_//SD1//'
filename='1_N'
load_dir=''
n_item=6
mixed_length=[1,]
trainset_ratio=[1.,]
repeat=False

l=len(mixed_length)
itemlist = list(range(1, n_item + 1))

totalset, trainset, testset = {},{},{}
for id in range(l):
    Seq=Gen_Seq(itemlist,mixed_length[id],repeat=repeat)
    totalset.update({'l'+str(mixed_length[id]):Seq})
    tr, te = balanced_select(Seq, int(len(Seq) * trainset_ratio[id]), n_item)
    trainset.update({'l'+str(mixed_length[id]):tr});testset.update({'l'+str(mixed_length[id]):te})

#save sequence set with respect to loading_dir
data={'totalset':totalset,'trainset':trainset,'testset':testset}
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not bool(load_dir):
    save_obj(data,out_dir+filename)
else:
    loadfile=load_obj(load_dir)
    for key in ['totalset','trainset','testset']:
        loadfile[key].update(data[key])
    save_obj(loadfile,out_dir+filename)





