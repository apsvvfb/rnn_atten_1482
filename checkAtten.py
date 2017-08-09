import numpy as np
import argparse
import sys
import h5py
import json

def readAttenFile(outpath,epoch,startb,batch_size,typen):
	attenfile="%s/epoch%d_from%d_batchsize%d.h5" %(outpath,epoch,startb,batch_size)
	if typen == "test":
		attenfile="%s/epoch%d_from%d_batchsize%d_test.h5" %(outpath,epoch,startb,batch_size)

	with h5py.File(attenfile,'r') as hf:
        	data = hf.get('atten_weight')
	        attenweights = np.array(data) #batch x shot_num x event_num
	return attenweights

def main(params):
	#typen="train"
	typen="test"
	if typen == "train":
		infile="randseq_train_atten.h5"
		total_sample_num = 20000
	elif typen == "test":
		infile="randseq_test_atten.h5"
                total_sample_num = 10000
	epoch_num = 10
	batch_size = 200
	shot_num = 30
	event_num = 10
	outpath = "./_atten_weight"
	with h5py.File(infile,'r') as hf:
		data = hf.get('mean')
		means = np.array(data)	#total_sample_num x shot_num
		data = hf.get('label')
		labels = np.array(data)
	startbs = [ bi*batch_size for bi in range(total_sample_num/batch_size) ]
	for startb in startbs:
                atten_weights = np.zeros((epoch_num,batch_size,shot_num,event_num))
		for epoch in range(epoch_num):
	                atten_weights[epoch] = readAttenFile(outpath,epoch+1,startb+1,batch_size,typen)
		for i_smp in range(startb,startb+batch_size):
			label = labels[i_smp]
			print "\n\n#sample %d, label %d" %(i_smp+1,label)
			for i_shot in range(shot_num):
				print "\nvideolabel %d" %(label)
				print "shot %d, shot_label %d" %(i_shot+1,means[i_smp][i_shot])
				for i_epoch in range(epoch_num):
					print "epoch %d" %(i_epoch+1)
					print atten_weights[i_epoch][i_smp-startb][i_shot]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_json', default='../data/vqa_data_prepro.json', help='input json file contains imgnames for train and test')
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed input parameters:'
	print json.dumps(params, indent = 2)
	main(params)
