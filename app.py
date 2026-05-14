import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    st.title("Login Page")
    username= st.text_input("username")
    password= st.text_input("password", type="password")
    if st.button("Login"):
        if username == "admin" and password =="admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")





# Features and target
X = df[[
    "followers",
    "avg_likes",
    "avg_comments",
    "posts_per_week",
    "engagement_rate"
]]

y = df["fake"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

if not st.session_state.logged_in:
    login()
    st.stop()

# Streamlit UI
st.title("Fake Follower Detection AI")
st.sidebar.title("About Project")

uploaded_file = st.file_uploader(
    "Upload Influencer Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    df = pd.read_csv("data/fake_follower.csv")

st.sidebar.info(
    """
    This AI project detects whether an influencer
    likely has fake followers using machine learning.

    Technologies Used:
    - Python
    - SQL
    - Machine Learning
    - Streamlit
    """
)

st.write("Enter influencer details below:")

followers = st.number_input("Followers")
likes = st.number_input("Average Likes")
comments = st.number_input("Average Comments")
posts = st.number_input("Posts Per Week")
engagement = st.number_input("Engagement Rate")

if st.button("Predict"):

    new_data = [[
        followers,
        likes,
        comments,
        posts,
        engagement
    ]]

    result = model.predict(new_data)
    confidence= model.predict_proba(new_data)
    

    if result[0] == 1:
        st.error("Status: Fake Influencer Detected")
        st.write(f"Confidence: {confidence[0][1]*100:.2f}%")
    else:
        st.success("Status: Genuine Influencer")
        st.write(f"Confidence: {confidence[0][0]*100:.2f}%")

st.subheader("Influencer Statistics")

stats = {
    "Followers": followers,
    "Likes": likes,
    "Comments": comments,
    "Posts": posts,
    "Engagement": engagement
}

fig, ax = plt.subplots()

ax.bar(stats.keys(), stats.values())

st.pyplot(fig)


st.subheader("Input Summary")

col1, col2 = st.columns(2)

col1.metric("Followers", followers)
col1.metric("Likes", likes)

col2.metric("Comments", comments)
col2.metric("Engagement", engagement)

