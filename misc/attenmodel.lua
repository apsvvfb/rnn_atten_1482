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

	local H_tmp, A, x_i, h_i, xh_i, a_i
        for i = 1, shot_num*2,2 do
                table.insert(inputs, nn.Identity()())
		table.insert(inputs, nn.Identity()())
                x_i = inputs[i]
		h_i = inputs[i+1]
                xh_i = nn.JoinTable(2){x_i, h_i}          --batch x (feat_dim + hiddensize)
                a_i = mlp(xh_i)                           --batch x event_num     
                if i == 1 then
                        H_tmp = h_i
                        A = a_i
                else
                        H_tmp = nn.JoinTable(2){H_tmp, h_i}     --batch x (hidden_size*shot_num)
                        A =  nn.JoinTable(1){A, a_i}            --(batch*shot_num) x event_num
                end
        end
	
        local H = nn.View(1,-1):setNumInputDims(1)(H_tmp)     -- batch x 1 x (hidden_size*shot_num) 

	A:annotate{name ='AW_1', description = 'attention weight'}
        --
	local A_rep = nn.Replicate(hidden_size,2)(A)          --(batch*shot_num) x hidden_size x event_num
        local A_res = nn.Reshape(shot_num, batch_size, hidden_size, event_num)(A_rep)
        local A_trans = nn.Transpose({1,2})(A_res)            --batch_size x shot_num x hidden_size x event_num
        local A_c = nn.Reshape(batch_size, shot_num*hidden_size, event_num)(A_trans)
	--]]


        --local A_c = nn.Reshape(batch_size, shot_num*hidden_size, event_num)(nn.Transpose({1,2})(nn.Reshape(shot_num, batch_size, hidden_size, event_num)(nn.Replicate(hidden_size,2)(A))))

        local out = nn.SoftMax()(nn.Reshape(batch_size, event_num)(nn.MM(){H,A_c}))

        table.insert(outputs,out)
	--table.insert(outputs,A)


        return(nn.gModule(inputs, outputs))

end

return attenmodel