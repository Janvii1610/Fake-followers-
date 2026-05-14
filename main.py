import pandas as pd
from sqlalchemy import create_engine,text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/fake_follower.csv") 

engine = create_engine("sqlite:///database/fake_follower.db")

df.to_sql("influencers", con=engine, if_exists="replace", index=False)


with engine.connect() as conn: result = conn.execute(text("""SELECT influencer_name, engagement_rate FROM influencers WHERE engagement_rate < 1"""))
for row in result: print(row)

X = df[[
    "followers",
    "avg_likes",
    "avg_comments",
    "posts_per_week",
    "engagement_rate"
]]

y = df["fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

followers = int(input("Enter followers: "))
likes = int(input("Enter average likes: "))
comments = int(input("Enter average comments: "))
posts = int(input("Enter posts per week: "))
engagement = float(input("Enter engagement rate: "))

new_data = [[
    followers,
    likes,
    comments,
    posts,
    engagement
]]

result = model.predict(new_data)

print("\n--- Prediction Result ---")

if result[0] == 1:
    print("Status: Fake Influencer Detected")
    print("Reason: Low engagement compared to follower count")
else:
    print("Status: Real Influencer")
    print("Reason: Healthy engagement rate")
 