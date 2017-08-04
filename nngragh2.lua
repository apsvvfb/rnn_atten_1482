require 'nn'
require 'nngraph'

feat_dim = 10
hidden_size = 5
class_num = 8
batch_size = 2
input_dim = feat_dim + hidden_size
shot_num=3
mid_dim = 4

x1 = torch.randn(batch_size,feat_dim)
x2 = torch.randn(batch_size,feat_dim)
x3 = torch.randn(batch_size,feat_dim)

        mlp = nn.Sequential();  -- make a multi-layer perceptron
        mlp:add(nn.Linear(input_dim, mid_dim))
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(mid_dim, class_num))

        local h1 = nn.Linear(feat_dim, hidden_size):forward(x1) -- batch x hiddensize
        local h2 = nn.Linear(feat_dim, hidden_size):forward(x2)
        local h3 = nn.Linear(feat_dim, hidden_size):forward(x3)

        xh1 = nn.JoinTable(2):forward{x1, h1}   --batch x (feat_dim + hiddensize)
        a1 = mlp:forward(xh1)                   --batch x class_num 
        xh2 = nn.JoinTable(2):forward{x2, h2}
        a2 = mlp:forward(xh2)
        xh3 = nn.JoinTable(2):forward{x3, h3}
        a3 = mlp:forward(xh3)
	print(#h1,#a1)

        H_tmp = nn.JoinTable(2):forward{h1,h2,h3}               --batch x (hidden_size*shot_num)
        H = nn.View(1,-1):setNumInputDims(1):forward(H_tmp)     -- batch x 1 x (hidden_size*shot_num) 
        A = nn.JoinTable(1):forward{a1,a2,a3}                   --(batch*shot_num) x class_num
        A_rep = nn.Replicate(hidden_size,2):forward(A)          --(batch*shot_num) x hidden_size x class_num
        A_res = nn.Reshape(shot_num, batch_size, hidden_size, class_num):forward(A_rep)
        A_trans = nn.Transpose({1,2}):forward(A_res)            --batch_size x shot_num x hidden_size x class_num
        A_fin = nn.Reshape(batch_size, shot_num*hidden_size, class_num,1):forward(A_trans)
        A_split = nn.SplitTable(1):forward(A_fin)               --class_num { batch_size x (shot_num*hidden_size) x 1 }

	print(A_fin)
	print(A_split)
--[[
        scores = {}
        for i = 1, class_num do
                score_tmp = nn.Reshape(batch_size,1):forward(nn.MM():forward({H, A_split[i]}))  --batch x 1 x 1 -> batch x 1
                table.insert(scores,score_tmp)
        end
        local out_tmp = nn.JoinTable(2):forward(scores)
        local out = nn.SoftMax():forward(out_tmp)
	print(out)--]]
