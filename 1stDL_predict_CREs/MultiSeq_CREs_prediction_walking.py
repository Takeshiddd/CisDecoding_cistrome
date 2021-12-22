#!/usr/bin/env python
# coding: utf-8
# Input should be fasta format without linebreaks
# This code outputs sequences of prediction (confidence) values for each TF-binding (with a trained model) per promoter sequences.

import numpy as np
import os
import re
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model,np_utils
from keras.models import Model, load_model

from optparse import OptionParser
usage = "USAGE: this.py [-f] [-m] [-w] [-b] [-o]"
parser = OptionParser(usage=usage)
parser.add_option("-f", dest="fasta", action="store",help="File path to fasta")
parser.add_option("-m", dest="trainmodel", action="store", help="File path to HDF5 or H5 model to predict TF-binding sites")
parser.add_option("-w", dest="walk", action="store", help="walk bp size")
parser.add_option("-b", dest="bin", action="store", help="bin bp size")
parser.add_option("-o", dest="out", action="store", help="output file name")


(opt, args) = parser.parse_args()

fasta = opt.fasta
trainmodel = opt.trainmodel
walk = opt.walk
bin = opt.bin
out = opt.out

os.system("awk '!/>/ {print}' " + fasta + " > prepped-seq.fa")
os.system("awk '/>/ {print}' " + fasta + " > prepped-OTU.txt")
outfile =open(out, "w")
fas = open("prepped-seq.fa",'r')
fas1 = fas.readlines()

def dna2num(dna):
    if dna.upper() == "A":
        return 0
    elif dna.upper() == "T":
        return 1
    elif dna.upper() == "G":
        return 2
    else:
        return 3

def num2dna(num):
    if num == 0:
        return "A"
    elif num == 1:
        return "T"
    elif num == 2:
        return "G"
    else:
        return "C"
    
def dna2array(DNAstring):
    numarr = []
    length = len(DNAstring)
    for i in range(0, length):
        num = dna2num(DNAstring[i:i+1])
        if num >= 0:
            numarr.append(num)
    return numarr

def array2dna(numarr):
    DNAstring = []
    length = numarr.shape[0]
    for i in range(0, length):
        dna = num2dna(numarr[i].argmax())
        DNAstring.append(dna)
    DNAstring = ''.join(DNAstring)
    return DNAstring

model = load_model(trainmodel,compile=False)

for line in fas1:
    line=line.rstrip()
    length = len(line)
    n = 0
    sub = []
    while length >= int(bin) + (n*int(walk)):
        sub.append(line[n*int(walk):int(bin)+ n*int(walk)])
        n += 1
    X = []
    for line2 in sub:
        OneHotArr = np.array([np.eye(4)[dna2array(line2)]])
        X.extend(OneHotArr)
    X = np.array(X)
    X = np.reshape(X,(-1, int(bin), 4, 1))
        
    predictions = model.predict(X)
    orig_names = []
    for img in X:
        orig_names.append(array2dna(img))
    
    for i in range(len(orig_names)):
        pred = predictions[i]
        outfile.write(str(pred[1]) + "\t")
    outfile.write("\n")
    
fas.close()
outfile.close()
os.system("paste prepped-OTU.txt " + out + " > wOTU-" + out)
os.system("rm prepped-OTU.txt")
os.system("rm prepped-seq.fa")

