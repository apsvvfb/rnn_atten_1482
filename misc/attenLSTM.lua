require 'nn'
require 'os'
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
	self.atten = attenmodel.attenfunc(mlp_input_dim, mlp_mid_dim, self.event_num, self.feat_dim, self.hidden_size, self.shot_num, self.batch_size)
	
end

function layer:getModulesList()
	return {self.atten}
end

function layer:parameters()
	local p1,g1 = self.atten:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	
	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end

	return params, grad_params
end

function layer:training()
	self.atten:training()
end

function layer:evaluate()
	self.atten:evaluate()
end

function layer:updateOutput(input)
	self.softmax_scores = self.atten:forward(input)

	return {self.softmax_scores}
end

function layer:updateGradInput(input, gradOutput)

	local dummy = self.atten:backward(input, gradOutput)

	self.gradInput = dummy
	return self.gradInput
end
