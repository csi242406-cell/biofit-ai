# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("BioFit AI: 웨어러블 생체 데이터 기반 피로 예측 앱")

st.write("심박수, 수면시간, 걸음 수, 운동시간 데이터를 바탕으로 피로 위험도를 예측합니다.")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("업로드한 데이터")
    st.dataframe(df)

    required_cols = ["평균심박수", "최대심박수", "수면시간", "걸음수", "운동시간", "주관적피로도"]

    if all(col in df.columns for col in required_cols):
        df = df.dropna()

        # 피로도 기준으로 위험도 라벨 생성
        def make_label(fatigue):
            if fatigue <= 3:
                return 0   # 정상
            elif fatigue <= 6:
                return 1   # 주의
            else:
                return 2   # 위험

        df["위험도"] = df["주관적피로도"].apply(make_label)

        X = df[["평균심박수", "최대심박수", "수면시간", "걸음수", "운동시간"]]
        y = df["위험도"]

        if len(df) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.subheader("모델 정확도")
            st.write(f"{acc:.2f}")

            st.subheader("새로운 데이터 입력")

            avg_hr = st.number_input("평균 심박수", min_value=40, max_value=200, value=75)
            max_hr = st.number_input("최대 심박수", min_value=60, max_value=220, value=130)
            sleep = st.number_input("수면시간", min_value=0.0, max_value=12.0, value=6.5)
            steps = st.number_input("걸음 수", min_value=0, max_value=50000, value=8000)
            exercise = st.number_input("운동시간(분)", min_value=0, max_value=300, value=40)

            new_data = pd.DataFrame({
                "평균심박수": [avg_hr],
                "최대심박수": [max_hr],
                "수면시간": [sleep],
                "걸음수": [steps],
                "운동시간": [exercise]
            })

            prediction = model.predict(new_data)[0]

            st.subheader("예측 결과")

            if prediction == 0:
                st.success("정상 상태로 예측됩니다.")
            elif prediction == 1:
                st.warning("주의 상태로 예측됩니다. 휴식과 수면 관리가 필요합니다.")
            else:
                st.error("위험 상태로 예측됩니다. 무리한 운동을 피하고 충분한 회복이 필요합니다.")

            st.subheader("변수 중요도")
            importance = pd.DataFrame({
                "변수": X.columns,
                "중요도": model.feature_importances_
            }).sort_values(by="중요도", ascending=False)

            st.bar_chart(importance.set_index("변수"))

        else:
            st.warning("AI 모델 학습을 위해 최소 10개 이상의 데이터가 필요합니다.")

    else:
        st.error("CSV 파일에 다음 열이 모두 필요합니다:")
        st.write(required_cols)

else:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")
