from gene_seq import *
import argparse
import numpy as np
from rw import *
import os

parser=argparse.ArgumentParser()
parser.add_argument("--out_dir",help="seq data out put filepath",type=str,default='')
parser.add_argument("--n_item",help="number of item",type=int,default=6)
parser.add_argument("--mixed_length",help="upper limit of sequence length",type=int,default=3)
parser.add_argument("--trainset_ratio",help="ratio of training set",type=float,nargs='+')#default=[1,0.6,0.4]

args=parser.parse_args()


n_item = args.n_item;mixed_length = args.mixed_length;trainset_ratio = tuple(args.trainset_ratio)
itemlist = list(range(1, n_item + 1))
totalset, trainset, testset = [], [], []
for id in range(mixed_length):
    Seq = fixedLen_seq(itemlist, id+1)
    totalset.append(Seq)
    tr, te = balanced_select(Seq, int(len(Seq) * trainset_ratio[id]), n_item)
    trainset.append(tr);testset.append(te)
totalset = tuple(totalset);trainset = tuple(trainset);testset = tuple(testset)

out_dir=args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.savez(out_dir+"//Seq_set.npz",totalset=totalset,trainset=trainset,testset=testset)
param={'n_item':n_item,'mixed_length':mixed_length,'trainset_ratio':trainset_ratio}
w_dict(out_dir+"//param.txt",param)





