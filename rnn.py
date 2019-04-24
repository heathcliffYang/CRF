import numpy as np
import torch.nn as nn
import torch
from basic_tool import *

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wifi_size, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.wifi_w = nn.Linear(wifi_size, hidden_size)
        self.i2m = nn.Linear(input_size + hidden_size, input_size + hidden_size)
        self.m_relu = nn.ReLU()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden): 
        combined = torch.cat((input, hidden), 0).to(self.device)
        combined = self.i2m(combined)
        combined = self.m_relu(combined)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def h0_init(self, wifi):
        input_wifi = torch.FloatTensor(-wifi).to(self.device)
        h0 = self.wifi_w(input_wifi)
        return h0

class RL_RNN(object):
    def __init__(self, input_size, hidden_size, output_size, wifi_size, lr, device, lambda1):
        self.net = RNN(input_size, hidden_size, output_size, wifi_size, device).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.Loss = nn.CrossEntropyLoss()
        self.device = device
        self.lr_sche = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1, last_epoch=-1)

    def forward_5_rounds(self, wifi, coordinate, radius):
        h0 = self.net.h0_init(wifi)
        action_history = torch.zeros([5,5])
        for i in range(5):
            input = torch.FloatTensor(np.concatenate((coordinate, np.array([radius])))).to(self.device)
            action, h0 = self.net.forward(input, h0)
            action_history[i] = action
            coordinate, radius = take_action(torch.argmax(action).item(), coordinate, radius)
        return action_history, coordinate, radius

    def back_propagation(self, action_history, action_label):
        # print("action_history", action_history)
        loss = self.Loss(action_history, torch.LongTensor(action_label))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("Pred", torch.argmax(action_history, dim=1))
        return loss

    def peek_weights(self):
        for param in self.net.parameters():
            print(param)

    def grading(self, action_history, action_label):
        grading_sheet = np.zeros((5,))
        for i in range(5):
            if (torch.argmax(action_history[i]).item() == action_label[i]):
                grading_sheet[i] = 1
        return grading_sheet