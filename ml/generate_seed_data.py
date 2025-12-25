import random
import pandas as pd

# -------- Productive components --------
productive_verbs = [
    "Studied", "Solved", "Practiced", "Worked on",
    "Completed", "Revised", "Prepared", "Built"
]

subjects = [
    "DSA", "DBMS", "Operating Systems", "Machine Learning",
    "Computer Networks", "Maths", "Python", "Web Development"
]

contexts = [
    "for exam",
    "for placement preparation",
    "for college assignment",
    "for project work",
    "in the morning",
    "for 2 hours",
    "seriously",
    "with focus"
]

productive_platforms = [
    "on LeetCode",
    "on Codeforces",
    "using textbooks",
    "from lecture notes",
    "from online course"
]

# -------- Non-productive components --------
non_productive_verbs = [
    "Scrolling", "Watching", "Playing",
    "Chatting on", "Browsing"
]

apps = [
    "Instagram", "YouTube", "WhatsApp",
    "Snapchat", "Facebook"
]

games = [
    "PUBG", "Free Fire", "Valorant",
    "GTA", "Minecraft"
]

non_contexts = [
    "randomly",
    "without purpose",
    "for time pass",
    "late night",
    "casually"
]

data = []

# Generate productive samples
for _ in range(250):
    sentence = f"{random.choice(productive_verbs)} {random.choice(subjects)} " \
               f"{random.choice(productive_platforms)} {random.choice(contexts)}"
    data.append([sentence, 1])

# Generate non-productive samples
for _ in range(250):
    if random.random() < 0.5:
        sentence = f"{random.choice(non_productive_verbs)} {random.choice(apps)} " \
                   f"{random.choice(non_contexts)}"
    else:
        sentence = f"Playing {random.choice(games)} {random.choice(non_contexts)}"
    data.append([sentence, 0])

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("ml/productivity_dataset_500.csv", index=False)

print("✅ Generated realistic 500-sample productivity dataset")
import pandas as pd
df = pd.read_csv("ml/productivity_dataset_500.csv")
print(df.shape)
df.sample(10)
