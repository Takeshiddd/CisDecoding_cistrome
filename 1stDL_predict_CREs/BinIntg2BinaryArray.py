
import pandas as pd
import os
import re
import numpy as np

from optparse import OptionParser

usage = "USAGE: this.py [-i] [-b] [-t] [-o]"
parser = OptionParser(usage=usage)
parser.add_option("-i", dest="input", action="store",help="File path to input")
parser.add_option("-b", dest="bin_size", action="store", help="bin size (int)")
parser.add_option("-t", dest="thres", action="store", help="confidence threshold (0-1)")
parser.add_option("-o", dest="out", action="store", help="output file name")

(opt, args) = parser.parse_args()

input = opt.input
bin_size = opt.bin_size
out = opt.out
threshold = opt.thres

df = pd.read_csv(input,
                 sep ="\t", header=None, index_col=0)
df_bin = df.groupby((np.arange(len(df.columns)) // int(bin_size)) + 1, axis=1).sum()

df_bool = df_bin >= float(threshold)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 10000
bidf = df_bool*1
bidf.to_csv(out, header=None, sep="\t")

