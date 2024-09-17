# To split the large csv into smaller ones
import os
import pandas as pd

# load the data
df = pd.read_csv('AudioSetPaths.csv', sep = '\t')

# create the 'splits' folder if it doesn't exist
if not os.path.exists('unbalanced_splits_audioset'):
    os.makedirs('unbalanced_splits_audioset')

# split the data
split_size = len(df) // 10  # this will give the size for each split
for i in range(10):
    start = i * split_size
    end = (i + 1) * split_size if i < 9 else None  # to include possible leftovers for the last file
    subset = df[start:end]
    subset.to_csv(f'unbalanced_splits_audioset/split_{i+1}.csv', index=False)  # write subset to csv

