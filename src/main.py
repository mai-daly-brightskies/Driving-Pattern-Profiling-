import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Attack_Classifier_Model.attack_page_classifier import attak_classifer
from Driver_Classifer_Model.driver_page_classifer import driver_model

# Define page navigation variable
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to render the home page
def home_page():
    st.title("🛡️🔒Driving Pattern Profiling & Car Hacking Classification")
    
    st.markdown("""
    <style>
        .intro {
            font-size: 18px;
            line-height: 1.8;
            margin-bottom: 20px;
        }
        .objective {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
        }
        .summary-title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .icon-container {
            text-align: center;
            margin: 20px 0;
        }
        .icon-container span {
            font-size: 50px;
            color: #4CAF50;
            margin: 0 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="intro">
    Although many anti-theft technologies are implemented, auto-theft is still increasing. Security vulnerabilities of cars can be exploited for auto-theft by neutralizing anti-theft systems. This keyless auto-theft attack will increase as cars adopt computerized electronic devices more. To detect auto-theft efficiently, we propose a driver verification method that analyzes driving patterns using measurements from sensors in the vehicle.
    </div>

    <div class="intro">
    This project uses machine learning to profile and classify driving patterns, extracting valuable insights from real-world vehicular data. Additionally, with modern vehicles' extensive connectivity, protecting in-vehicle networks from cyber-attacks, particularly on the Controller Area Network (CAN) protocol, has become essential. Due to its lack of security features, CAN is vulnerable to attacks like message injection.
    </div>

    <div class="summary-title">Objectives</div>
    <div class="intro">
    <ul>
        <li>Develop machine learning models capable of profiling and classifying driving patterns using real-world vehicular data.</li>
        <li>Explore feature extraction techniques and data preprocessing methods to capture the characteristics of driving behavior.</li>
        <li>Evaluate the performance of the developed models in accurately identifying and categorizing different driving patterns.</li>
        <li>Create a web app to visualize the results of the model.</li>
    </ul>
    </div>

    <div class="summary-title">Functionalities</div>
    <div class="intro">
    This application provides two main functionalities:
    <ul>
        <li><strong>Attack Classification</strong>: Detect and classify various types of cyber-attacks on in-vehicle networks.</li>
        <li><strong>Driver Classification</strong>: Profile and classify driving patterns to verify driver identity and detect abnormal behaviors.</li>
    </ul>
    </div>

    <div class="intro">
    Choose a functionality below to explore more:
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Attack Model"):
            st.session_state.page = 'attack_model'
            st.experimental_rerun()
    with col2:
        if st.button("Go to Driver Classifier"):
            st.session_state.page = 'driver_classifier'
            st.experimental_rerun()

# Render the appropriate page based on the navigation variable
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'attack_model':
    attak_classifer()
elif st.session_state.page == 'driver_classifier':
    driver_model()
