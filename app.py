import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

from ml_app import run_ml_app


def main():
    st.title('Object detecion')


    # 사이드바 메뉴
    menu= ['Home','ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write('물체를 인식해줍니다.')
        st.write('왼쪽의 사이드바에서 ML을 선택하세요.')
        st.image('data/test4.jpg')

    elif choice =='ML':
        run_ml_app()

if __name__ == '__main__':
    main()