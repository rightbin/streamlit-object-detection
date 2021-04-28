import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
import tempfile




def run_ml_app():
    
    st.title('COCO Dataset')

    st.write('다음은 odject-detection을 하기 전의 영상입니다.')
    st.write('다음의 비디오는 Pixabay에서 다운로드 받았습니다.')

    video_file = open('data/India.mp4', 'rb')
    video_bytes = video_file.read()
        
    st.video(video_bytes)

    st.write('다음은 odject-detection을 한 후의 영상입니다.')

    

    video_result = open('data/output.mp4','rb')

    video2 = video_result.read()
            
    st.video(video2)

    return