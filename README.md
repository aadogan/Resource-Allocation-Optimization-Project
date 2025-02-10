**Resource Allocation Optimization Project**

This project aims to optimize the fair and efficient distribution of critical resources (e.g., hospital beds or vaccines). Synthetic patient data will be generated, and reinforcement learning (RL) methods will be used to model and optimize resource allocation decisions. Additionally, techniques such as SHAP or LIME will be applied to explain the model's decisions, and an interactive interface will be developed to facilitate human-AI interaction.

--------------------------------

**Project Objectives**

**1-)** Synthetic Data Generation: Create synthetic patient data without using real patient data.

**2-)** Data Preparation: Enhance the generated synthetic data with additional variables (e.g., Severity and Priority) to make it suitable for modeling.

**3-)** Optimization with Reinforcement Learning: Approach the resource allocation problem as an RL problem and train an agent that balances fairness and efficiency.

**4-)** Explaining Model Decisions: Analyze the model's decisions using explainability methods to make the decision-making process transparent.

**5-)** Interactive Dashboard Development: Build a user-friendly interface to enhance human-AI collaboration.

-------------------------------

**Required Libraries and Setup**

The following Python libraries are required to run the project:

Python 3.10.X-3.11.X

llama-cpp-python (requires **cmake** and **git** to be installed on your machine)

Transformers

Torch

NumPy

Pandas

SHAP

LIME

Streamlit

Gym

-----------------------------------

**synthetic_data_generation.py**

Purpose:

Used to generate synthetic patient data. It leverages the Llama model to create detailed and structured patient profiles. This data serves as the foundation for model training and simulations without using real patient data.

Key Features:

 - Data Generation:

Produces synthetic patient data using a specially designed prompt.

- Data Processing:

Structures the generated text by parsing it into a structured dataset.

- Data Saving:

Saves the resulting data in CSV format as synthetic_patient_data.csv.

---------------------------------

**prepare_data.py**

Purpose:

Processes the generated synthetic patient data to make it suitable for modeling. Specifically, it adds Severity and Priority variables for each patient and saves the processed data.

Key Features:

- Data Loading:

Loads the synthetic_patient_data.csv file or uses patient.csv if it exists.

- Adding New Variables:

Calculates severity and priority scores for patients based on their age and other attributes.

- Data Saving:

Saves the processed data to processed_patient_data.csv.

--------------------------------------


**environment.py**

Purpose:

Defines a custom reinforcement learning environment to model the resource allocation problem. It uses the OpenAI Gym library to implement the environment's dynamics and reward function.

Key Features:

- Reward Mechanism:

Calculates the reward function by considering fairness and efficiency criteria.

- Resource Management:

Tracks the available resources and terminates the episode when resources are depleted.

---------------------------------------



**model_explanation.py**

Purpose:

Used to explain and analyze the decisions of the trained model. It leverages the SHAP library to visualize the importance of different features in the model's decision-making process.

Key Features:

- Model Loading:

Loads the trained DQN model.

- SHAP Value Calculation:

Calculates SHAP values for the selected action.

- Visualization:

Visualizes the model's decisions using SHAP summary plots, making it easier to interpret and understand the model's behavior.

-----------------------------------------




**streamlit_app.py**

Purpose:

Provides an interactive interface for the project. It uses the Streamlit library to create a user-friendly dashboard.

----------------------------------------
**Shap Visual**

<img width="539" alt="Figure 2025-02-07 223547" src="https://github.com/user-attachments/assets/d614b008-400a-4494-a8b5-9f4abf0b056c" />

--------------------------
**Dashboard V1**

[dashboard · Streamlit.pdf](https://github.com/user-attachments/files/18734222/dashboard.Streamlit.pdf)

----------------------------

**Dashboard V2**

![Ekran görüntüsü ](https://github.com/user-attachments/assets/28925162-cfad-468a-9f79-c9b864a0ee31)

![Ekran görüntüsü ](https://github.com/user-attachments/assets/7767922d-e9f2-4bed-bde4-5d89c21a71da)


--











