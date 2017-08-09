import numpy as np
import sys
from random import randint
import h5py

feat_dim = 100 #1024
#batch_size = 5 
shot_num = 30 #30
labelnum = 10
#batch_num = total_sample_num / batch_size
outfile_train = "randseq_train.h5"
outfile_test = "randseq_test.h5"

sigma = 0.1 #standard deviation
###############train
total_sample_num = 10000
out = np.zeros((total_sample_num, shot_num, feat_dim))
labels = np.zeros(total_sample_num)
for i in range(total_sample_num):
	randi = randint(1, labelnum) 
	mu = randi #mean
	labels[i] = randi
	out[i] = np.random.normal(mu, sigma, (shot_num,feat_dim) )

with h5py.File(outfile_train, 'w') as hf:
	hf.create_dataset('feature', data=out)
	hf.create_dataset('label', data=labels)

###########test
total_sample_num = 10000
out = np.zeros((total_sample_num, shot_num, feat_dim))
labels = np.zeros(total_sample_num)
for i in range(total_sample_num):
        randi = randint(1, labelnum) 
        mu = randi #mean
        labels[i] = randi
        out[i] = np.random.normal(mu, sigma, (shot_num,feat_dim) )

with h5py.File(outfile_test, 'w') as hf:
        hf.create_dataset('feature', data=out)
        hf.create_dataset('label', data=labels)
