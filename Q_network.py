import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.active1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(n_hidden, 32)
        self.active2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.active1(x) # ReLU # activation
        x = self.fc2(x)
        x = self.active2(x)
        actions_value = self.fc3(x)
        
        return actions_value


class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state, epsilon, i, t, it):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if (np.random.uniform() < epsilon):
            action = np.random.randint(0, self.n_actions)
        else: 
            actions_value = self.eval_net(x, i, t, it) # 以現有 eval net 得出各個 action 的分數
#             print("act v", actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # 挑選最高分的 action
#             print("act ! ", action)
#             print("Sum ", torch.sum(actions_value, dim = 1))

        return action
    

    def store_transition(self, state, action, reward, next_state):
        # 打包 experience
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def BP(self, state, next_state, reward, right, i, t, it):
        q_eval = self.eval_net(torch.unsqueeze(torch.FloatTensor(state),0), i, t, it)
        q_target = torch.LongTensor([right])
#         print("loss q", ho)
#         print(q_target)
        loss = self.loss_func(q_eval, q_target)
#         loss = self.loss_func(torch.unsqueeze(ho, 0), torch.unsqueeze(torch.tensor(right, dtype=torch.long),0))
#         q_eval = self.eval_net(torch.FloatTensor(state))
#         q_next = self.target_net(torch.FloatTensor(next_state)).detach()
#         q_target = reward + self.gamma * q_next
        # print("loss :", loss)
#         loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def learn(self):
        # 隨機取樣 batch_size 個 experience
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_eval, ho = self.eval_net(b_state).gather(1, b_action) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        loss = self.loss_func(ho, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def load(self, model_path):
        self.eval_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(self.eval_net.state_dict())
