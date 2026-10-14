
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/heart.csv")

print("Dataset Shape:", data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nTarget Distribution:")
print(data["target"].value_counts().sort_index())

print("\nStatistical Summary:")
print(data.describe())

data["target"].value_counts().sort_index().plot(
    kind="bar",
    rot=0
)

plt.title("Distribution of Heart Disease Target Classes")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.show()
