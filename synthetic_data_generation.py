import os
import pandas as pd
from llama_cpp import Llama

# Modelinizin yolunu belirtin
model_path = r"C:\........\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Llama modelini yükleyin
llm = Llama(model_path=model_path)

# Sentetik hasta verilerini oluşturmak için fonksiyonlar
def generate_patient_data():
    system_prompt = (
        "<<SYS>>\n"
        "You are an assistant specialized in the medical field, speaking English, and generating detailed and structured patient data. "
        "While creating realistic and logical synthetic patient profiles, please adhere to ethical guidelines.\n"
        "<</SYS>>\n\n"
    )
    
    user_prompt = (
        "Please create **one** synthetic patient profile using the following format:\n\n"
        "**Patient ID**: [Automatically increasing number starting from 1]\n"
        "**Name**: [A realistic English name]\n"
        "**Age**: [An age between 0 and 100]\n"
        "**Gender**: [Male or Female]\n"
        "**Medical History**:\n"
        "  - **Complaints**: [Patient's current complaints]\n"
        "  - **Diagnoses**: [Previous and current diagnoses]\n"
        "  - **Allergies**: [Known allergies, if any]\n"
        "  - **Family History**: [Genetic diseases in the family]\n"
        "**Medications**: [Medications the patient is using]\n"
        "**Social History**: [Occupation, lifestyle, smoking/alcohol use]\n"
        "**Vital Signs**:\n"
        "  - **Blood Pressure**: [e.g., 120/80 mmHg]\n"
        "  - **Heart Rate**: [e.g., 72 bpm]\n"
        "  - **Respiratory Rate**: [e.g., 16 breaths/min]\n"
        "  - **Temperature**: [e.g., 36.8°C]\n"
        "**Laboratory Results**:\n"
        "  - **Blood Tests**: [Normal or abnormal values]\n"
        "  - **Imaging Results**: [Radiology results, if any]\n"
        "\n"
        "Please fill in the patient information logically and consistently and only give what is requested, add no own interpretation"
    )
    
    full_prompt = f"[INST] {system_prompt}{user_prompt} [/INST]"
    output = llm(
        full_prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        repeat_penalty=1.15,
        stop=["</s>", "[/INST]"]
    )
    return output["choices"][0]["text"].strip()

def parse_patient_data(text):
    data = {}
    lines = text.strip().split('\n')
    current_section = None
    for line in lines:
        if '**' in line:
            key_value = line.strip().split('**:', 1)
            if len(key_value) == 2:
                key = key_value[0].replace('**', '').strip()
                value = key_value[1].strip()
                data[key] = value
                current_section = key
            else:
                sub_key_value = line.strip().split(':', 1)
                if len(sub_key_value) == 2:
                    sub_key = sub_key_value[0].replace('-', '').strip()
                    sub_value = sub_key_value[1].strip()
                    if current_section and isinstance(data.get(current_section), dict):
                        data[current_section][sub_key] = sub_value
                    else:
                        data[current_section] = {sub_key: sub_value}
        else:
            sub_key_value = line.strip().split(':', 1)
            if len(sub_key_value) == 2:
                sub_key = sub_key_value[0].replace('-', '').strip()
                sub_value = sub_key_value[1].strip()
                if current_section and isinstance(data.get(current_section), dict):
                    data[current_section][sub_key] = sub_value
                else:
                    data[current_section] = {sub_key: sub_value}
    return data


patient_data = []
num_patients = 100  # Üretilecek hasta sayısı

for i in range(1, num_patients + 1):
    print(f"Generating patient {i}...")
    text = generate_patient_data()
    data = parse_patient_data(text)
    data['Patient ID'] = i
    patient_data.append(data)


df_patients = pd.json_normalize(patient_data)


columns_order = [
    'Patient ID', 'Name', 'Age', 'Gender',
    'Medical History.Complaints', 'Medical History.Diagnoses', 'Medical History.Allergies', 'Medical History.Family History',
    'Medications', 'Social History',
    'Vital Signs.Blood Pressure', 'Vital Signs.Heart Rate', 'Vital Signs.Respiratory Rate', 'Vital Signs.Temperature',
    'Laboratory Results.Blood Tests', 'Laboratory Results.Imaging Results'
]

df_patients = df_patients.reindex(columns=columns_order)


df_patients['Patient ID'] = df_patients['Patient ID'].astype(int)
df_patients['Age'] = pd.to_numeric(df_patients['Age'], errors='coerce').fillna(0).astype(int)


df_patients.fillna('Unknown', inplace=True)


df_patients.to_csv("synthetic_patient_data.csv", index=False, encoding='utf-8-sig')


print(df_patients.head())
