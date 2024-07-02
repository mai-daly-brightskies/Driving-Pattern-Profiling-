# Driving Pattern Profiling & Car Hacking Classification

## Overview
Although many anti-theft technologies are implemented, auto-theft is still increasing. Also, security vulnerabilities of cars can be used for auto-theft by neutralizing anti-theft system. This keyless auto-theft attack will be increased as cars adopt computerized electronic devices more. To detect autotheft efficiently, we propose the driver verification method that analyzes driving patterns using measurements from the sensor in the vehicle.

This project uses machine learning to profile and classify driving patterns, extracting valuable insights from real-world vehicular data. Additionally, with modern vehicles' extensive connectivity, protecting in-vehicle networks from cyber-attacks, particularly on the Controller Area Network (CAN) protocol, has become essential. Due to its lack of security features, CAN is vulnerable to attacks like message injection.

## Objectives
- Develop machine learning models capable of profiling and classifying driving patterns using real-world vehicular data.
- Explore feature extraction techniques and data preprocessing methods to capture the characteristics of driving behavior.
- Evaluate the performance of the developed models in accurately identifying and categorizing different driving patterns.
- Create a web app to visualize the results of the model.

## Dataset

### First Phase
The project utilizes real-world vehicular data collected from onboard sensors. The data encompasses approximately 23 hours of driving time, covering a total distance of approximately 46 kilometers on a round-trip route between Korea University and SANGAM World Cup Stadium. This dataset includes driving data from 10 distinct drivers, categorized into classes A to J.

### Second Phase
We provide car-hacking datasets which include DoS attacks, fuzzy attacks, spoofing the drive gear, and spoofing the RPM gauge. These datasets were constructed by logging CAN traffic via the OBD-II port from a real vehicle while message injection attacks were performed. Each dataset contains 300 intrusions of message injection, with each intrusion lasting 3 to 5 seconds, and each dataset includes a total of 30 to 40 minutes of CAN traffic.

## Methodology

### Phase 1: Driving Pattern Profiling
1. **Data Preprocessing:** Clean and preprocess the data using Pandas and NumPy to handle noise and missing values.
2. **Feature Extraction:** Use feature extraction techniques to capture the most important features of driving behavior.
3. **Model Development:** Develop machine learning models using scikit-learn and PyToch to profile and classify driving patterns.
4. **Model Evaluation:** Evaluate the performance of the models using metrics like accuracy and recall.
5. **Result Visualization:** Create a web app using Streamlit to visualize the results.

### Phase 2: Car Hacking Classification
1. **Data Preprocessing:** Process the CAN traffic data using Pandas and NumPy to identify and segment the attack instances.
2. **Feature Extraction:** Extract features that distinguish normal traffic from attack traffic.
3. **Model Development:** Develop machine learning models using scikit-learn to classify and detect different types of car hacking attempts.
4. **Model Evaluation:** Evaluate the performance of the models using metrics like accuracy, precision, recall, and F1-score.
5. **Result Visualization:** Create a web app using Streamlit to visualize the results.

## Working Steps
### Phase 1: Driving Pattern Profiling
(Compelete Later)


### Phase 2: Car Hacking Classification
Our project began with the collection of various files containing both normal messages and specific attack messages. The first step involved merging all these files together to create a dataset.

#### Data Cleaning
To ensure data integrity, we followed specific cleaning steps:
- Dropped any rows where the DLC (Number of data bytes) was not equal to 8, as these rows had null targets.

#### Exploratory Data Analysis (EDA)
During the EDA phase, several important insights were revealed:
- Most `can_id` values had only one flag, though some had two.
- The most prevalent attack across most `can_id` values was the fuzzy attack.
- Overall, the frequency of all types of attacks was approximately equal.
- Each bit had its own distinct effect on the target, indicating the complexity and diversity of the data.

#### Data Preprocessing
Addressing the issue of unbalanced data was crucial. We achieved balance by taking a random sample of normal messages and combining them with all messages from each attack type. This resulted in a dataset where each class represented approximately 20% of the total data.

#### Data Splitting
The balanced dataset was then split into training, validation, and test sets to ensure robust model evaluation.

#### Model Development
We applied logistic regression to classify the data. The model's performance on the validation data yielded impressive results:

**Classification Report on Validation Data:**
```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98     94215
           1       0.95      0.94      0.95     78684
           2       0.94      0.91      0.92     73231
           3       1.00      1.00      1.00    104361
           4       1.00      1.00      1.00     95467

    accuracy                           0.97    445958
   macro avg       0.97      0.97      0.97    445958
weighted avg       0.97      0.97      0.97    445958
```

The model was then tested on unseen data, producing similarly strong results:

**Classification Report on Test Data:**
```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98    117385
           1       0.95      0.94      0.95     98316
           2       0.94      0.91      0.92     90866
           3       1.00      1.00      1.00    131397
           4       1.00      1.00      1.00    119483

    accuracy                           0.97    557447
   macro avg       0.97      0.97      0.97    557447
weighted avg       0.97      0.97      0.97    557447
```

These results underscore the model's robustness and accuracy in attack types.

## Tools and Libraries

<p align="left"> <a href="https://www.w3schools.com/cs/" target="_blank" rel="noreferrer"> 
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://numpy.org/doc/stable/_static/numpylogo.svg" alt="NumPy" width="80" height="40"/> </a> <a href="https://mlflow.org/" target="_blank" rel="noreferrer"> <img src="https://th.bing.com/th/id/OIP.OsB57V0FPteixDBD_BBN4gHaCt?w=321&h=128&c=7&r=0&o=5&pid=1.7" alt="MlFlow" width="60" height="40"/> </a> <a href="https://optuna.org/" target="_blank" rel="noreferrer"> <img src="https://th.bing.com/th/id/R.03ca3566557285f85f4f6fb9c0b99ff4?rik=zrr1Q1bqVILT1w&pid=ImgRaw&r=0" alt="Optuna" width="100" height="60"/> </a> <a href="https://streamlit.io/" target="_blank" rel="noreferrer"> <img src="https://cdn.knoji.com/images/logo/streamlit.jpg?fit=contain&trim=true&flatten=true&extend=25&width=1200&height=630" alt="Streamlit" width="80" height="80"/> </a> 

## Recommendations for Future Work
1. **Expand Data Collection:** Include more diverse driving conditions and a larger number of drivers to improve model generalization.
2. **Enhance Feature Engineering:** Investigate additional features and sensor data to capture more detailed driving behaviors.
3. **Advanced Models:** Explore deep learning techniques for better performance in both driving pattern profiling and car hacking detection.
4. **Real-Time Analysis:** Develop systems for real-time detection and mitigation of cyber-attacks on in-vehicle networks.
5. **Collaborative Research:** Partner with automotive manufacturers and cybersecurity experts to refine models and address emerging threats.
6. **Continuous Evaluation:** Implement continuous learning to adapt models based on new data and evolving driving patterns and attack methods.

## Installation and Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mai-daly-brightskies/Driving-Pattern-Profiling-.git
   cd Driving-Pattern-Profiling-

   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
