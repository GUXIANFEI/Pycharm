import torch
import torch.utils.data as Data

Batch_size = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch.dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
loader = Data.DataLoader(
    dataset = torch.dataset,
    batch_size= Batch_size,
    shuffle=True,
    num_workers=2,
)

for epoch in range(3):
    for step, (batch_x,batch_y) in enumerate(loader):
        #training
        print ('Epoch: ', epoch,'|Step: ',step,'|batch x: ',batch_x.numpy(),
                '|batch y: ',batch_y.numpy())



#hyper parameter

LR = 0.1
Batch_size = 32
Epoch = 12

x = torch.unsqueeze(torch.linspace(-1,1,1000),dim = 1)
y = x.pow(2)+0.2*torch.rand(x.size())

torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=Batch_size,
    shuffle=True,
    num_workers=2,
)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net(1,10,1)
opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)

