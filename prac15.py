import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# 2. TITLE
# --------------------------------
st.title("🎓 Student Performance Prediction System")
st.write(
    "This machine learning application predicts student academic "
    "performance using student information."
)
st.divider()

# --------------------------------
# 3. GENERATE SAMPLE DATASET
# --------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)

    data = pd.DataFrame({
        "Study_Hours": np.random.randint(1, 10, 200),
        "Attendance": np.random.randint(50, 101, 200),
        "Previous_Score": np.random.randint(30, 101, 200),
        "Sleep_Hours": np.random.randint(4, 10, 200)
    })

    data["Final_Score"] = (
        data["Study_Hours"] * 4
        + data["Attendance"] * 0.25
        + data["Previous_Score"] * 0.35
        + data["Sleep_Hours"] * 1.5
        + np.random.normal(0, 5, 200)
    ).clip(0, 100)

    return data


df = generate_data()

# --------------------------------
# 4. DATASET DISPLAY
# --------------------------------
st.header("📊 Student Dataset")
st.write("Sample dataset used for machine learning:")
st.dataframe(df.head(20), use_container_width=True)
st.write("Dataset Shape:", df.shape)

# --------------------------------
# 5. DATA VISUALIZATION
# --------------------------------
st.header("📈 Data Visualization")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["Final_Score"], kde=True, ax=ax)
    ax.set_title("Distribution of Final Scores")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df, x="Study_Hours", y="Final_Score", ax=ax
    )
    ax.set_title("Study Hours vs Final Score")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------
# 6. MACHINE LEARNING MODEL
# --------------------------------
st.header("🤖 Machine Learning Model")

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours"
]
X = df[features]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Model R² Score", f"{r2:.2%}")
col2.metric("Mean Absolute Error", f"{mae:.2f}")

# --------------------------------
# 7. USER INPUT
# --------------------------------
st.divider()
st.header("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Daily Study Hours", min_value=1, max_value=12, value=5
    )
    attendance = st.slider(
        "Attendance Percentage", min_value=0, max_value=100, value=80
    )

with col2:
    previous_score = st.slider(
        "Previous Exam Score", min_value=0, max_value=100, value=70
    )
    sleep_hours = st.slider(
        "Daily Sleep Hours", min_value=1, max_value=12, value=7
    )

# --------------------------------
# 8. PREDICTION
# --------------------------------
if st.button("Predict Student Performance"):
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score, sleep_hours]],
        columns=features
    )

    prediction = float(np.clip(model.predict(input_data)[0], 0, 100))

    st.subheader("🎯 Prediction Result")
    st.metric("Predicted Final Score", f"{prediction:.2f} / 100")

    if prediction >= 75:
        st.success("Excellent Performance!")
    elif prediction >= 50:
        st.info("Average Performance. Keep Improving!")
    else:
        st.warning("Needs Improvement!")

    st.progress(int(prediction))

# --------------------------------
# 9. FOOTER
# --------------------------------
st.divider()
st.caption(
    "Student Performance Prediction System | "
    "Developed using Python, Streamlit and Scikit-learn"
)
