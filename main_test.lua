require 'rnn'
require 'os'
require 'nn'
require 'optim'
--require 'cutorch'
--require 'cunn'
require 'misc.attenLSTM'
-- model parameters
attOpt = {} 
attOpt.feat_dim = 20
attOpt.hidden_size = 10
attOpt.batch_size = 7 
attOpt.shot_num = 5
attOpt.event_num = 10
attOpt.mlp_mid_dim = 5
sgd_params = {
   learningRate = 0.01,
}

model_attenLSTM = nn.attenLSTM(attOpt)
--model_attenLSTM:cuda()
params, gradParams = model_attenLSTM:getParameters()
--local criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)
local criterion = nn.ClassNLLCriterion()
---------------input
local targets = {}
local outputs, err = {}, 0
local inputs = {}
for seq = 1,attOpt.shot_num do
	local input_i = torch.randn(attOpt.batch_size, attOpt.feat_dim) --:cuda()
	table.insert(inputs, input_i)
end
targets = torch.Tensor(attOpt.batch_size):fill(3) --:cuda()
--------------forward
outputs = unpack(model_attenLSTM:forward(inputs))
print(outputs)
loss = criterion:forward(outputs, targets)
print(loss)
---------------backward
gradOutputs = criterion:backward(outputs, targets)
model_attenLSTM:backward(inputs, gradOutputs)
--------------update1
--[[
model_attenLSTM:updateParameters(0.01)
model_attenLSTM:forget()
model_attenLSTM:zeroGradParameters()
--]]
--------------update2
feval = function(params_new)
        -- copy the weight if are changed
        if params ~= params_new then
                params:copy(params_new)
        end
        return loss, gradParams
end
_, fs = optim.sgd(feval, params, sgd_params)
model_attenLSTM:forget()
model_attenLSTM:zeroGradParameters()

--------------test
outputs = unpack(model_attenLSTM:forward(inputs))
print(criterion:forward(outputs, targets))
