import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_coco_app():
    
    st.title('COCO Dataset')

    st.write('COCO Dataset은 Object Detection, Segmentation, Keypoint Detection을 제공하는 MICROSOFT가 제작한 API입니다.')

    st.write('관련된 영상을 참조해주세요.')

    st.video("https://www.youtube.com/watch?v=h6s61a_pqfM")

    st.write('그럼 자율주행에 odject-detection이 왜 중요한 것일까요?')
    st.write('다음의 영상을 참고하시면 이해하시는데 도움이 되실 것입니다.')

    st.video("https://www.youtube.com/watch?v=rnV1gpyFh5A")


    return

    