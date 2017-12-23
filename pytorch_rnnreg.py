import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

Lr = 0.01
Time_step = 40
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state = self.rnn(x,h_state)
        out = []
        for time_step in range(r_out.size(1)):
            out.append(self.out(r_out[:,time_step,:]))
        return torch.stack(out,dim=1),h_state


rnn = RNN()
print rnn

optimizer = torch.optim.Adam(rnn.parameters(),lr = Lr)
loss_func = nn.MSELoss()

h_state = None
plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(120):
    start,end = step*np.pi,(step+1)*np.pi
    steps = np.linspace(start,end,Time_step,dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction,h_state = rnn(x,h_state)
    h_state = Variable(h_state.data)        #this step is important

    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()
