require 'nngraph'
require 'nn'
require 'rnn'

local attenmodel = {} 
function attenmodel.attenfunc(input_dim, mid_dim, event_num, feat_dim, hidden_size, shot_num, batch_size)
        local inputs = {}
        local outputs = {}

        mlp = nn.Sequential();  -- make a multi-layer perceptron
        mlp:add(nn.MaskZero(nn.Linear(input_dim, mid_dim),1))
        mlp:add(nn.MaskZero(nn.Tanh(),1))
        mlp:add(nn.MaskZero(nn.Linear(mid_dim, event_num),1))

        for i = 1, shot_num do
                table.insert(inputs, nn.Identity()())
                local x_i = inputs[i]
                --local h_i = nn.Linear(feat_dim, hidden_size)(x_i)
                local h_i = nn.FastLSTM(feat_dim, hidden_size):maskZero(1)(x_i)
                local xh_i = nn.JoinTable(2){x_i, h_i}          --batch x (feat_dim + hiddensize)
                local a_i = mlp(xh_i)                           --batch x event_num     
                if i == 1 then
                        H_tmp = h_i
                        A = a_i
                else
                        H_tmp = nn.JoinTable(2){H_tmp, h_i}     --batch x (hidden_size*shot_num)
                        A =  nn.JoinTable(1){A, a_i}            --(batch*shot_num) x event_num
                end
        end

        --H_tmp = nn.JoinTable(2){h1,h2,h3}               --batch x (hidden_size*shot_num)
        H = nn.View(1,-1):setNumInputDims(1)(H_tmp)     -- batch x 1 x (hidden_size*shot_num) 
        --A = nn.JoinTable(1){a1,a2,a3}                   --(batch*shot_num) x event_num

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

return attenmodel
