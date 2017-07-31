require 'nn'
require 'os'
--require 'cutorch'
--require 'cunn'

local utils = require 'misc.utils'
local attenmodel = require 'misc.attenmodel'
local layer, parent = torch.class('nn.attenLSTM','nn.Module')

function layer:__init(opt) 
	parent.__init(self)
	self.feat_dim = utils.getopt(opt, 'feat_dim') 
    	self.hidden_size = utils.getopt(opt, 'hidden_size')
	self.shot_num = utils.getopt(opt, 'shot_num')
	self.event_num =  utils.getopt(opt, 'event_num')
	self.batch_size = utils.getopt(opt, 'batch_size')
	self.mlp_mid_dim = utils.getopt(opt, 'mlp_mid_dim')
	
	local mlp_input_dim = self.feat_dim + self.hidden_size
	local mlp_mid_dim = self.mlp_mid_dim

	--self.rnn = nn.Sequencer(nn.FastLSTM(self.feat_dim, self.hidden_size):maskZero(1))
	self.rnn = nn.SeqLSTM(self.feat_dim, self.hidden_size)
	self.rnn.maskzero = true
	self.atten = attenmodel.attenfunc(mlp_input_dim, mlp_mid_dim, self.event_num, self.feat_dim, self.hidden_size, self.shot_num, self.batch_size)
end

function layer:getModulesList()
	return {self.rnn,self.atten}
end

function layer:parameters()
	local p1,g1 = self.rnn:parameters()
	local p2,g2 = self.atten:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
        for k,v in pairs(p2) do table.insert(params, v) end
	
	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end
        for k,v in pairs(g2) do table.insert(grad_params, v) end

	return params, grad_params
end

function layer:training()
	self.rnn:training()
	self.atten:training()
end

function layer:evaluate()
	self.rnn:evaluate()
	self.atten:evaluate()
end

function layer:updateOutput(input)
	local hidden = self.rnn:forward(input)

	self.input_and_hidden = {}
	for i = 1, self.shot_num do
		table.insert(self.input_and_hidden, input[i])
		table.insert(self.input_and_hidden, hidden[i])
	end	

	local softmax_scores = self.atten:forward(self.input_and_hidden)

	return {softmax_scores}
end

function layer:updateGradInput(input, gradOutput)
	local d_inputhidden = self.atten:backward(self.input_and_hidden, gradOutput)
	local d_hidden = {}
	for i = 1, self.shot_num*2,2 do
		table.insert(d_hidden, d_inputhidden[i+1])
	end
	d_hidden_ = nn.JoinTable(1):forward(d_hidden):reshape(self.shot_num,self.batch_size,self.hidden_size)
	if input:type() == 'torch.CudaTensor' then
		d_hidden_ = d_hidden_:cuda()
	end
	local dummy = self.rnn:backward(input,d_hidden_)
	self.gradInput = dummy
	return self.gradInput
end
