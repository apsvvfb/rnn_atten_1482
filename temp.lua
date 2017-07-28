require 'os'
require 'cutorch'
require 'cunn'
require 'nn'
require 'nngraph'
torch.manualSeed(1)
input = nn.Identity()()
L1 = nn.ReLU()(nn.Linear(3, 1)(input))
net = nn.Sequential()
net:add(L1)
g = nn.gModule({input}, {L1})
x = torch.randn(3)
g:forward(x)
g:cuda()
g:forward(x:cuda())
g:float()
g:forward(x)

os.exit()



batch_size = 3
shot_num = 6
class_num = 5
hidden_size = 4

module = nn.Replicate(hidden_size,2)
module2 = nn.Reshape(shot_num,batch_size,hidden_size,class_num)
module3 = nn.Transpose({1,2})
module4 = nn.Reshape(batch_size,shot_num*hidden_size,class_num)

input=torch.randn(batch_size*shot_num,class_num)

out=module:forward(input)
out2 = module2:forward(out)
out3 = module3:forward(out2)
out4 = module4:forward(out3)
print(input)
--print(out)
--print(out2[1],out2[2])
--print(out3[1],out3[2],out3[3])
print(out4[1],out4[2],out4[3])

