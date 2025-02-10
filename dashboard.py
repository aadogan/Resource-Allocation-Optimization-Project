import streamlit as st
import torch
import numpy as np
import pandas as pd
from training import DQN
from environment import ResourceAllocationEnv


data_path = r"C:\.....\Belgeler\1pythondsy\new_project\processed_patient_data.csv"
df_patients = pd.read_csv(data_path)



env = ResourceAllocationEnv(df_patients)

# Eğitilmiş modeli yükleyin
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

st.title("Kaynak Tahsisi RL Simülasyonu")

if st.button("Simülasyonu Başlat"):
    state = env.reset()
    total_reward = 0
    while True:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state, reward, done, _ = env.step(action)
        
        
        patient = env.patient_data.iloc[env.current_patient_index - 1]
        
        st.write(f"**Hasta ID: {patient['Patient ID']}**")
        st.write(f"Yaş: {patient['Age']}, Ciddiyet: {patient['Severity']:.2f}")
        st.write(f"Eylem (Kaynak Tahsisi): {'Evet' if action == 1 else 'Hayır'}")
        st.write(f"Ödül: {reward:.2f}")
        st.write("---")
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    st.success(f"Toplam Ödül: {total_reward:.2f}")
