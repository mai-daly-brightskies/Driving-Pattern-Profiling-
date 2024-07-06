import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Attack_Classifier_Model.attack_page_classifier import attak_classifer
from Driver_Classifer_Model.driver_page_classifer import driver_model
from Driver_Classifer_Model import driver_page_classifer


# Define page navigation variable
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Homepage
if st.session_state.page == 'home':
    st.title("Welcome to the Classification Demo")
    st.markdown("""
    This application provides two main functionalities:
    - Attack Classification
    - Driver Classification
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Attack Model"):
            st.session_state.page = 'attack_model'
            st.experimental_rerun()
    with col2:
        if st.button("Go to Driver Classifier"):
            st.session_state.page = 'driver_classifier'
            st.experimental_rerun()


# Attack Model page
if st.session_state.page == 'attack_model':
    attak_classifer()
    
# Driver Classifier page
if st.session_state.page == 'driver_classifier':
    driver_model()
    