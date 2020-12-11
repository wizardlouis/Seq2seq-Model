import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--filepath',help='load filepath dir',type=str,default='')
args=parser.parse_args()


with open(args.filepath,'r',encoding='utf-8') as f:
  content = f.read()

command=content.split(';')
length=len(command)
for i in range(length):
    os.system(command[i]+' &')
