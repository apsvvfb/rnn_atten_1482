require 'nngraph'
require 'nn'
require 'rnn'
local attenmodel = {} 
function attenmodel.attenfunc(input_dim, mid_dim, event_num, feat_dim, hidden_size, shot_num, batch_size)
	local inputs = {}
        local outputs = {}

        local mlp = nn.Sequential();  -- make a multi-layer perceptron
        mlp:add(nn.MaskZero(nn.Linear(input_dim, mid_dim),1))
        mlp:add(nn.MaskZero(nn.Tanh(),1))
        mlp:add(nn.MaskZero(nn.Linear(mid_dim, event_num),1))

        local H, A, x_i, h_i, xh_i, a_i
        for i = 1, shot_num*2,2 do
                table.insert(inputs, nn.Identity()())
                table.insert(inputs, nn.Identity()())
                x_i = inputs[i]				--batch x feat_dim
                h_i = inputs[i+1]			--batch x hidden_size
                xh_i = nn.JoinTable(2){x_i, h_i}	--batch x (feat_dim + hiddensize)
                a_i = mlp(xh_i):annotate{name ='mlp_layer', description = 'mlp layer'}   --batch x event_num     
                if i == 1 then
                        H = h_i
			A = a_i --A and a_i use same memory, so when a_i change, A will change.
			--use two lines below because of problem of memory copy 
                        B=nn.SplitTable(1)(A)
                        A=nn.Reshape(batch_size, event_num)(nn.JoinTable(1)(B))
                else
                        H = nn.JoinTable(1,1){H, h_i}	--batch x (hidden_size*shot_num)
                        A = nn.JoinTable(1,1){A, a_i}	--batch x (event_num*shot_num)
                end
        end
	local H2 = nn.Reshape(shot_num,hidden_size,true)(H)	--batch x shot_num x hidden_size
	local A2 = nn.Reshape(shot_num,event_num,true)(A)	--batch x shot_num x event_num
        A2:annotate{name ='AW_2', description = 'attention weight'}
	local A3 = nn.Transpose({1,2}):setNumInputDims(2)(A2)	--batch x event_num x shot_num

	local normalize = nn.ParallelTable()
        for i = 1, batch_size do
        	normalize:add(nn.Normalize(1))
        end
	local m = nn.Sequential()
        m:add(nn.SplitTable(1))
        m:add(normalize)
	local A4 = nn.Reshape(batch_size, event_num, shot_num)(nn.JoinTable(1)(m(A3)))			--batch x event_num x shot_num

	local H_bar = nn.MM(){A4,H2}	--batch x event_num x hidden_size
	local reduceDim = nn.ParallelTable()
	for i = 1, event_num do
		reduceDim:add(nn.Linear(hidden_size, 1))
	end
	m = nn.Sequential()
	m:add(nn.SplitTable(1,2))
	m:add(reduceDim)
	local out = nn.JoinTable(1,1)(m(H_bar))			--batch x event_num

	local softscore = nn.LogSoftMax()(out)

        table.insert(outputs,softscore)
        attenmodule = nn.gModule(inputs, outputs)
        return(attenmodule)
end

return attenmodel
