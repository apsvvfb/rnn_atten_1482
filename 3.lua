require 'rnn'
require 'os'
require 'nn'
require 'optim'

printout=1
debug=0
rnn= nn.Sequential()
      :add(nn.LSTM(4,10))
      :add(nn.Linear(10,2))
      :add(nn.LogSoftMax())
----------------------don't change the order-----------------------
rnn2=rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
params, gradParams = rnn:getParameters()
rnn3=rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
params, gradParams = rnn:getParameters()
rnn4=rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
params, gradParams = rnn:getParameters()
--------------------------------------------------------------------
inputs = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
a = torch.randn(3,4)
inputs = {a,a,a}
--targets= {torch.randn(3,1), torch.randn(3,1), torch.randn(3,1)}
targets = {}
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
nStep=3
------------------------------------------ single step
-------------------------optim
print("########################single step: optim")
sgd_params = {
   learningRate = 0.01,
}
outputs, loss = {}, 0
criterion = nn.ClassNLLCriterion()
for step=1,3 do
   outputs[step] = rnn:forward(inputs[step])
   print(outputs[step])
   --loss = loss + criterion:forward(outputs[step], targets[step])
end
os.exit()
gradOutputs, gradInputs = {}, {}
rnn:zeroGradParameters()
for step=3,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end
feval = function(params_new)
        -- copy the weight if are changed
        if params ~= params_new then
                params:copy(params_new)
        end
        -- select a training batch
        --local inputs, targets = nextBatch()

        -- reset gradients (gradients are always accumulated, to accommodate
        -- batch methods)
        --dl_dx:zero()

        -- evaluate the loss function and its derivative with respect to x, given a mini batch

        --local outputs = seq:forward(inputs)
        --local loss_x = criterion:forward(outputs, targets)
        --local gradOutputs = criterion:backward(outputs, targets)
        --local gradInputs = seq:backward(inputs, gradOutputs)
        return loss, gradParams
end
gradParams1=gradParams:clone()
params1=params:clone()
_, fs = optim.sgd(feval,params, sgd_params)
rnn:forget()
rnn:zeroGradParameters()

if printout==1 then
	print(rnn:forward(inputs[1]))
end
----------------------updataParameters
print("########################single step:updataParameters")
outputs, loss = {}, 0 
criterion = nn.ClassNLLCriterion()
for step=1,3 do
   outputs[step] = rnn3:forward(inputs[step])
   if printout==1 and step == 1 then
        print(outputs[step])
   end     
   loss = loss + criterion:forward(outputs[step], targets[step])
end     

gradOutputs, gradInputs = {}, {}
rnn3:zeroGradParameters()
for step=3,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn3:backward(inputs[step], gradOutputs[step])
end

rnn3:updateParameters(0.01)
rnn3:forget()
rnn3:zeroGradParameters()
if printout==1 then
        print(rnn3:forward(inputs[1]))
end

