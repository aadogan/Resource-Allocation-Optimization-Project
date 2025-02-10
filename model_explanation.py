import torch
import numpy as np
import pandas as pd
import shap
from training import DQN
from environment import ResourceAllocationEnv

# İşlenmiş hasta verilerini yükleyin
df_patients = pd.read_csv("processed_patient_data.csv")


env = ResourceAllocationEnv(df_patients)

# Eğitilmiş modeli yükleyin
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

# SHAP için durumları hazırlayın
states = []
for i in range(min(100, env.num_patients)):
    state = env.get_patient_features(i)
    states.append(state)

states = np.array(states)

# Model tahmin fonksiyonu
def model_predict_action1(state_numpy):
    state_tensor = torch.FloatTensor(state_numpy)
    with torch.no_grad():
        q_values = model(state_tensor)
        q_value_action1 = q_values[:, 1]
    return q_value_action1.numpy()

# SHAP açıklayıcıyı oluşturun
explainer = shap.KernelExplainer(model_predict_action1, states)
shap_values = explainer.shap_values(states)


print(f"States shape: {states.shape}")
print(f"shap_values shape: {shap_values.shape}")


shap.initjs()
shap.summary_plot(shap_values, states, feature_names=['Age', 'Severity'])
