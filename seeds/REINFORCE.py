import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(environment):
    env = gym.make(environment)
    pi = Policy()
    model_parameters = count_parameters(pi)
    score = 0.0
    print_interval = 20
    average = 0.0
    for n_epi in range(5000):
        s = env.reset()[0]
        done = False
        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, _, _ = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
        pi.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            average = score/print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, average))
            score = 0.0
            if average >=  500:
                return n_epi, model_parameters
    env.close()
    return n_epi, model_parameters