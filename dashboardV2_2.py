import streamlit as st
import torch
import pandas as pd
from training import DQN
from environment import ResourceAllocationEnv


data_path = r"C:\.......\new_project\processed_patient_data.csv"
df_patients = pd.read_csv(data_path)

st.title("Kaynak Tahsisi RL Simülasyonu")

# Senaryo ve Müdahale Ayarları
scenario = st.sidebar.selectbox("Senaryo Seçin", ["Normal", "Acil Durum", "Aşı Kıtlığı"])

# Orijinal veriden bir kopya oluşturup senaryoya göre modifiye ediyoruz
df_patients_sim = df_patients.copy()
if scenario == "Acil Durum":
    # Örneğin, hastaların ciddiyetini %20 artırıyoruz
    df_patients_sim["Severity"] = df_patients_sim["Severity"] * 1.2
elif scenario == "Aşı Kıtlığı":
    # Farklı bir senaryoda ciddiyet etkisini düşürebiliriz
    df_patients_sim["Severity"] = df_patients_sim["Severity"] * 0.8


user_intervention = st.sidebar.checkbox("Model Kararına Müdahale Et", value=False)

# --- Modelin Yüklenmesi ---
# Modelin state ve action boyutunu öğrenmek için geçici bir ortam oluşturuyoruz
temp_env = ResourceAllocationEnv(df_patients_sim)
state_size = temp_env.observation_space.shape[0]
action_size = temp_env.action_space.n
del temp_env

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()


if "sim_started" not in st.session_state:
    st.session_state.sim_started = False
if "sim_done" not in st.session_state:
    st.session_state.sim_done = False

def init_simulation():
    st.session_state.env = ResourceAllocationEnv(df_patients_sim)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.total_reward = 0.0
    st.session_state.sim_done = False

# --- Simülasyonu Başlat ---
if not st.session_state.sim_started:
    if st.button("Simülasyonu Başlat"):
        st.session_state.sim_started = True
        init_simulation()
        st.experimental_rerun()

else:
    # Simülasyon tamamlandıysa
    if st.session_state.sim_done:
        st.success(f"Simülasyon tamamlandı! Toplam Ödül: {st.session_state.total_reward:.2f}")
        if st.button("Simülasyonu Yeniden Başlat"):
            st.session_state.sim_started = False
            st.experimental_rerun()
    else:
        # --- Simülasyon Devam Ediyor ---
        env = st.session_state.env
        state = st.session_state.state

        # Modelin önerdiği eylem
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = model(state_tensor)
        model_action = torch.argmax(q_values).item()
        model_action_str = "Evet" if model_action == 1 else "Hayır"
        st.write(f"**Modelin Kararı:** {model_action_str}")

        # Mevcut hasta bilgilerini göster
        # env.current_patient_index, hastanın sırasını tutan indekstir.
        patient = env.patient_data.iloc[env.current_patient_index - 1]
        st.write(f"**Hasta ID:** {patient['Patient ID']}")
        st.write(f"Yaş: {patient['Age']}, Ciddiyet: {patient['Severity']:.2f}")

        
        if user_intervention:
            default_index = 1 if model_action == 1 else 0
            user_choice = st.radio("Eylemi Seçin", options=["Evet", "Hayır"], index=default_index)
            final_action = 1 if user_choice == "Evet" else 0
            st.write(f"**Seçilen Eylem:** {'Evet' if final_action == 1 else 'Hayır'} (Kullanıcı müdahalesi)")
        else:
            final_action = model_action
            st.write(f"**Seçilen Eylem:** {model_action_str}")

        
        if st.button("Adımı Uygula"):
            next_state, reward, done, _ = env.step(final_action)
            st.session_state.total_reward += reward

            st.write("---")
            st.write(f"**Eylem Uygulandı:** {'Evet' if final_action == 1 else 'Hayır'}")
            st.write(f"**Adımda Alınan Ödül:** {reward:.2f}")

            st.session_state.state = next_state

            if done:
                st.session_state.sim_done = True

            st.experimental_rerun()