import gym
from gym import spaces
import numpy as np
import pandas as pd

class ResourceAllocationEnv(gym.Env):
    def __init__(self, patient_data):
        super(ResourceAllocationEnv, self).__init__()
        
        # Hasta verilerini yükleyin
        self.patient_data = patient_data.reset_index(drop=True)
        self.num_patients = len(self.patient_data)
        self.current_patient_index = 0
        
        # Eylem alanı: Kaynak tahsisi (0: Tahsis etme, 1: Tahsis et)
        self.action_space = spaces.Discrete(2)
        
        # Durum alanı: Hasta özellikleri (Age, Severity)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Kaynak kısıtlaması (örneğin, mevcut yatak sayısı)
        self.total_resources = 50
        self.available_resources = self.total_resources
        
        self.reset()
    
    def reset(self):
        self.current_patient_index = 0
        self.available_resources = self.total_resources
        patient_features = self.get_patient_features(self.current_patient_index)
        return patient_features
    
    def step(self, action):
        done = False
        info = {}
        
        
        patient = self.patient_data.iloc[self.current_patient_index]
        patient_priority = patient['Priority']
        
        
        reward = 0
        if action == 1 and self.available_resources > 0:
            self.available_resources -= 1
            reward = patient_priority  # Daha yüksek öncelikli hastalar daha fazla ödül getirir
        elif action == 0:
            reward = 0  # Kaynak tahsis edilmediğinde ödül yok
        else:
            reward = -1  # Kaynak yoksa ve tahsis etmeye çalışırsa ceza
        
        
        self.current_patient_index += 1
        if self.current_patient_index >= self.num_patients or self.available_resources <= 0:
            done = True
            next_state = np.zeros(self.observation_space.shape)
        else:
            next_state = self.get_patient_features(self.current_patient_index)
        
        return next_state, reward, done, info

    def get_patient_features(self, index):
        patient = self.patient_data.iloc[index]
        
        age = patient['Age'] / 100  
        severity = patient['Severity']
        features = np.array([age, severity], dtype=np.float32)
        return features

    @property
    def num_features(self):
        return 2
