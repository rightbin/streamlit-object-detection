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
from coco import run_coco_app

def main():
    
    # 사이드바 메뉴
    menu= ['Home','COCO','ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.title('Object detecion을 이용한 자율주행')
        
        st.write('객체 탐지(object detection)는 컴퓨터 비전과 이미지 처리와 관련된 컴퓨터 기술로써, 디지털 이미지와 비디오로 특정한 계열의 시맨틱 객체 인스턴스(예: 인간, 건물, 자동차)를 감지하는 일을 다룹니다.')

        st.write('다음과 같이 이미지의 물체를 탐지하는 과정을 Object detection이라 합니다.')

        video_file = open('data/OD.mp4', 'rb')
        video_bytes = video_file.read()
            
        st.video(video_bytes)
        
    elif choice =='COCO':
        run_coco_app()

    elif choice =='ML':
        run_ml_app()

    

if __name__ == '__main__':
    main()