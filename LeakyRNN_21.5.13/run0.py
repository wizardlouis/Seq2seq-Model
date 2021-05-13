import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--filepath',help='load filepath dir',type=str,default='')
parser.add_argument('--nohup',help='if use nohup',type=int,default=1)
args=parser.parse_args()


with open(args.filepath,'r',encoding='utf-8') as f:
  content = f.read()

command=content.split(';')
length=len(command)
if args.nohup==1:
    for i in range(length):
        os.system('nohup '+command[i]+' &')
else:
    for i in range(length):
        os.system(command[i]+' &')

