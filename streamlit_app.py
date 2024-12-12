import streamlit as st
import pandas as pd
import joblib

# 모델 및 데이터 로드
model = joblib.load('./models/trained_model.pkl')
data = pd.read_csv('./data/respiratory_data.csv')

# 심각도 맵핑
severity_map = {'낮음': 1, '중간': 2, '심각': 3}

# 사용자 입력
st.title('질병 예측 및 치료 추천 서비스')

st.sidebar.header('증상을 선택하고 심각도를 입력하세요.')
age = st.sidebar.slider('나이', 0, 100, 25)
gender = st.sidebar.selectbox('성별', ['여성', '남성'])
symptoms = st.sidebar.multiselect('증상을 선택하세요', ['기침', '가슴 답답함', '천명음', '호흡 곤란'])
severity = st.sidebar.selectbox('증상의 심각도를 선택하세요', ['낮음', '중간', '심각'])

# 데이터 준비
symptom_code = {'기침': 1, '가슴 답답함': 2, '천명음': 3, '호흡 곤란': 4}
gender_code = 0 if gender == '여성' else 1
severity_code = severity_map[severity]

input_data = pd.DataFrame([{
    '나이': age,
    '성별': gender_code,
    '증상_코드': symptom_code[symptoms[0]] if symptoms else 0,
    '상태_코드': severity_code
}])

# 예측
if st.button('질병 예측'):
    prediction = model.predict_proba(input_data)[0]
    predicted_disease = model.classes_[prediction.argmax()]
    st.write(f'예상 질병: {predicted_disease} (확률: {max(prediction)*100:.2f}%)')

    # 치료 방법 추천
    treatment = data[data['질환'] == predicted_disease]['치료 방법'].values[0]
    st.write(f'추천 치료 방법: {treatment}')
