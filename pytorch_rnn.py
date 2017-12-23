from torchvision import datasets
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


Epoch = 1
Batch_size = 64
Time_step = 28
input_size =28
Lr = 0.01

train_data = datasets.MNIST(
    root = './mnist',
    train = True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=Batch_size,
                               shuffle=True,
                               num_workers=2)

test_data = datasets.MNIST(
    root='./mnist',
    train = False,
    transform=torchvision.transforms.ToTensor()
)

test_x = Variable(test_data.test_data,volatile = True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)  #  x (batch,time_step,input_size)
        out = self.out(r_out[:,-1,:]) #(batch, time step, input)
        return out

rnn = RNN()
print rnn

optimizer = torch.optim.Adam(rnn.parameters(),lr = Lr)
loss_func = nn.CrossEntropyLoss()


for epoch in range(Epoch):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step %64 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y)/2000.
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
