require 'nn'
require 'nngraph'
function attenfunc(input_dim, mid_dim, event_num, feat_dim, hidden_size,shot_num,batch_size)
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local x1 = inputs[1]
	local x2 = inputs[2]
	local x3 = inputs[3]

        mlp = nn.Sequential();  -- make a multi-layer perceptron
        mlp:add(nn.Linear(input_dim, mid_dim))
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(mid_dim, event_num))

        local h1 = nn.Linear(feat_dim, hidden_size)(x1) -- batch x hiddensize
        local h2 = nn.Linear(feat_dim, hidden_size)(x2)
        local h3 = nn.Linear(feat_dim, hidden_size)(x3)

        xh1 = nn.JoinTable(2){x1, h1}   --batch x (feat_dim + hiddensize)
        a1 = mlp(xh1)                   --batch x event_num 
        xh2 = nn.JoinTable(2){x2, h2}
        a2 = mlp(xh2)
        xh3 = nn.JoinTable(2){x3, h3}
        a3 = mlp(xh3)

        H_tmp = nn.JoinTable(2){h1,h2,h3}               --batch x (hidden_size*shot_num)
        H = nn.View(1,-1):setNumInputDims(1)(H_tmp)     -- batch x 1 x (hidden_size*shot_num) 
        A = nn.JoinTable(1){a1,a2,a3}                   --(batch*shot_num) x event_num

        A_rep = nn.Replicate(hidden_size,2)(A)          --(batch*shot_num) x hidden_size x event_num
        A_res = nn.Reshape(shot_num, batch_size, hidden_size, event_num)(A_rep)
        A_trans = nn.Transpose({1,2})(A_res)            --batch_size x shot_num x hidden_size x event_num
        --A_fin = nn.Reshape(batch_size, shot_num*hidden_size, event_num,1)(A_trans)
        --A_split = nn.SplitTable(3)(A_fin)               --event_num { batch_size x (shot_num*hidden_size) x 1 }
	A_c = nn.Reshape(batch_size, shot_num*hidden_size, event_num)(A_trans)

	out_tmp = nn.Reshape(batch_size, event_num)(nn.MM(){H,A_c})
        local out = nn.SoftMax()(out_tmp)

        table.insert(outputs,out)
        return(nn.gModule(inputs, outputs))

end
---------------main func------------------
shot_num = 3 --number of input
feat_dim = 10
hidden_size = 5
event_num = 8
batch_size = 2
input_dim = feat_dim + hidden_size
mlp_mid_dim = 4
mlp_input_dim = feat_dim + hidden_size

local atten = attenfunc(mlp_input_dim, mlp_mid_dim, event_num, feat_dim, hidden_size, shot_num, batch_size)
x1 = torch.rand(batch_size,feat_dim)
--x2 = torch.rand(3,10)
x2 = x1:clone()
x3 = x1:clone()

out=atten:forward({x1, x2, x3})
print(out)
--print(out[1],out[2])
