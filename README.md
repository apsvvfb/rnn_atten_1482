# rnn_atten_1482
version 2.0
the test result is fine, however, this version has memory leak.
because lstm sequence will be calculated in nngraph.

version 3.0
the test result is fine.
I put the lstm module in the misc/attenLSTM instead of nngraph(misc/attenmodel).

version 4.0
the previous versions are wrong!! and shouldn't be used any more!!
I changed the misc/attenmodel.lua.
In the previous version, 
1.I wrongly calculated the output; 
2.I used softmax instead of logsoftmax(because I use ClassNLLCriterion, I should use logsoftmax)


how to prove the effectiveness of model:
0. generate data
	genTestSeq.py: without noise
	genTestSeq_noise.py : with noise
way1:(finished)
1. use "atten_test.lua" to test
2. use "checkAtten.py"

way2:(not start)
1. I want to change the misc/attenLSTM.lua : UpdateOutput(input) [set all attention weight to 1, then compare the results of using attention weight
