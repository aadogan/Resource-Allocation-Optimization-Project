import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from environment import ResourceAllocationEnv

# İşlenmiş hasta verilerini yükleyin
df_patients = pd.read_csv("processed_patient_data.csv")

env = ResourceAllocationEnv(df_patients)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# DQN modeli
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

model = DQN(state_size, action_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim parametreleri
num_episodes = 1000  # Daha hızlı sonuçlar için episd sayısını azaltabilirsiniz
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        state_tensor = torch.FloatTensor(state)
        if np.random.rand() <= epsilon:
            action = np.random.randint(action_size)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
        
        next_state, reward, done, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state)
        
        with torch.no_grad():
            q_values_next = model(next_state_tensor)
            max_q_value_next = torch.max(q_values_next)
            target_q_value = reward + gamma * max_q_value_next
        
        q_values_current = model(state_tensor)[action]
        
        loss = criterion(q_values_current, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")


torch.save(model.state_dict(), "dqn_model.pth")
