import streamlit as st
import numpy as np
import pickle

# 페이지 제목
st.title("질병 예측 서비스")

# 사용자 입력
symptoms = st.text_input("증상을 입력하세요 (예: coughing, wheezing):")
age = st.number_input("나이", min_value=0, max_value=120, step=1, value=30)
sex = st.selectbox("성별", options=["Male", "Female"])
nature = st.selectbox("증상의 심각도", options=["low", "medium", "high"])

# 모델 로드
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    if st.button("예측하기"):
        # 입력 데이터 처리
        symptoms_vectorized = vectorizer.transform([symptoms])
        sex_encoded = 0 if sex == "Male" else 1
        nature_encoded = ["low", "medium", "high"].index(nature)

        # 예측 데이터 구성
        input_data = np.hstack([symptoms_vectorized.toarray(), [sex_encoded], [nature_encoded]])
        prediction = model.predict(input_data)

        st.success(f"예측된 질병: {prediction[0]}")
except FileNotFoundError:
    st.error("모델 파일이 없습니다. 먼저 모델을 학습시키고 저장하세요.")
