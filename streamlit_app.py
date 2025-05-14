import streamlit as st
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# df = pd.read_csv('data/ACE_stat.csv', encoding='utf-8')
csv_data = st.secrets["project"]["data1"]
df = pd.read_csv(io.StringIO(csv_data), encoding = 'utf-8')
# st.dataframe(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

st.write ("## 당신의 운동 유형은?")

st.write ("운동 유형은 ACE 유전자형에 따라 II,ID,DD의 세가지로 나눌 수 있습니다.")
st.write ("다음 질문에 차례대로 답한 뒤 결과를 확인하세요!")

st.write ("#### Q1 일주일 평균 운동 시간")
exc = st.text_input("숫자만 작성해주세요 (예:5)", key="q1")
st.write ("#### Q2 하루 평균 수면 시간")
sleep = st.text_input("숫자만 작성해주세요 (예:8)", key="q2")
st.write ("#### Q3 PAPS 오래 달리기 등급")
long = st.selectbox("선택해주세요", ["1","2","3","4","5"],key="q3")
st.write ("#### Q4 PAPS 50m 달리기 등급")
short = st.selectbox("선택해주세요", ["1","2","3","4","5"],key="q4")


if st.button('예측하기'):
    try: 
        exc_value = float(exc)
        sleep_value = float(sleep)
        long_value = int(long)
        short_value = int(short)
        
        user_input = [[exc_value, sleep_value, long_value, short_value]]
        user_input_scaled = scaler.transform(user_input)

        prediction = knn.predict(user_input_scaled)
        genotype_labels = {0: "DD", 1: "ID", 2: "II"}
        selected_genotype = genotype_labels[prediction[0]]

        if prediction[0] == 0:
            st.success("당신은 DD형입니다 당신은 다른 사람에 비해 순발력이 필요한 종목에 강할 확률이 높습니다. 그러나 심폐지구력을 요하는 종목에는 어려움을 느낄 수 있습니다. 자신 있는 종목을 꾸준히 이어 나가고, 심폐지구력을 키울 수 있는 수영, 자전거 타기 등의 스포츠를 도전하여 순발력을 키우는 것을 추천합니다!")
        
        elif prediction[0] == 1:
            st.success("당신은 ID형입니다. 당신은 심폐지구력과 순발력 모두 적당한 균형을 이루고 있을 확률이 높습니다. 순발력을 키울 수 있는 배드민턴, 복싱 등과 심폐지구력을 키울 수 있는 수영, 자전거 등의 다양한 스포츠 종목에 도전하여 당신에게 맞는 종목을 찾아보세요!")

        elif prediction[0] == 2:
            st.success("당신은 II형입니다. 당신은 다른 사람에 비해 심폐지구력이 필요한 종목에 강할 확률이 높습니다. 그러나 순발력을 요하는 종목에는 어려움을 느낄 수 있습니다. 자신 있는 종목을 꾸준히 이어 나가고, 순발력을 필요로 하는 배드민턴, 단거리 달리기, 복싱 등의 스포츠를 도전하여 순발력을 키우는 것을 추천합니다!")

        st.write("#### 예측 확률:")
        for i, label in sorted(genotype_labels.items()):
            percent = round(prediction_proba[i] * 100, 2)
            st.write(f"- {label}형: {percent}%")    

    except ValueError:
        st.error("입력값이 유효하지 않습니다. 숫자만 입력해 주세요.")

        