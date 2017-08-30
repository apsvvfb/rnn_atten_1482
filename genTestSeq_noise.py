import numpy as np
import random
import sys
import h5py
import argparse
import json

from randomSeq import randomDistribution

def generateOneSeq(noise_num,shot_num,feat_dim,label,sigma):
	noise_idxes = random.sample(range(shot_num),noise_num)
        mu = label #mean
	#typeids_i = np.zeros(shot_num)
	mu_noise_i = np.ones(shot_num)*label
	sigma_noise_i = np.ones(shot_num)*0.1
	out_i = np.zeros((shot_num,feat_dim))
	for k in range(shot_num):
		if k in noise_idxes:
			#typeids_i[k], out_i[k] = randomDistribution(feat_dim)	
			foo=[1,-1]
		        PosOrNeg = np.random.choice(foo)
			mu_noise_i[k] = mu + PosOrNeg*random.randint(1,10)
			sigma_noise_i[k] = 0.1 + np.random.random() * (10 ** random.randint(0,3)) + np.finfo(float).eps
			out_i[k] = np.random.normal(mu_noise_i[k], sigma_noise_i[k], (1,feat_dim) )
		else:
			out_i[k] = np.random.normal(mu, sigma, (1,feat_dim) )
	#return out_i, typeids_i
	return out_i, mu_noise_i, sigma_noise_i

def generateSeq(outfile,total_sample_num,shot_num,label_num,feat_dim,noise_num,sigma):
	out = np.zeros((total_sample_num, shot_num, feat_dim))
	labels = np.zeros(total_sample_num)
	#typeids = np.zeros((total_sample_num,shot_num))
	mu_noise = np.zeros((total_sample_num,shot_num))
        sigma_noise = np.ones((total_sample_num,shot_num))*sigma

	if noise_num == 0:
		for i in range(total_sample_num):
			randi_label = random.randint(1, label_num) 
			mu = randi_label #mean
			labels[i] = randi_label
			out[i] = np.random.normal(mu, sigma, (shot_num,feat_dim) )
			mu_noise[i] = np.ones(shot_num) * labels[i]
	elif noise_num == -1:
		max_noise_num = shot_num / 2
		for i in range(total_sample_num):
			labels[i]= random.randint(1, label_num)
			out[i], mu_noise[i], sigma_noise[i] = generateOneSeq(random.randint(1, max_noise_num),shot_num,feat_dim,labels[i],sigma)
	elif noise_num > 0:
		for i in range(total_sample_num):
			labels[i] = random.randint(1, label_num)			
			out[i], mu_noise[i], sigma_noise[i] = generateOneSeq(noise_num,shot_num,feat_dim,labels[i],sigma)
	else:
		print "noise_num should be larger than/equal to -1"

	with h5py.File(outfile, 'w') as hf:
		hf.create_dataset('feature', data=out)
		hf.create_dataset('label', data=labels)
		#hf.create_dataset('typeid', data=typeids)
		hf.create_dataset('mean', data=mu_noise)
		hf.create_dataset('sigma',data=sigma_noise)

def main(params):
	outfile_train = (params['train_file'])
        outfile_test = (params['test_file'])

	train_num = 100000
	test_num = 10000
	shot_num = 30
	label_num = 10
	feat_dim = 100 #1024

	# 0: no noise. Values of all shots in the video will obey Gaussian distribution.
	#-1: random number for each sample 
	# i: a positive interger. i shots in the video will be generated randomly instead of obeying Gaussian distribution.
	noise_num = -1

	sigma = 0.1 #variance

	generateSeq(outfile_train,train_num, shot_num, label_num, feat_dim, noise_num, sigma)
	generateSeq(outfile_test, test_num, shot_num, label_num, feat_dim, noise_num, sigma)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file', default='randseq_train_atten.h5', help='output file for training')
	parser.add_argument('--test_file', default='randseq_test_atten.h5', help='')
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed input parameters:'
	print json.dumps(params, indent = 2)
	main(params)
