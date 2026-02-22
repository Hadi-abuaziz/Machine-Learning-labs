import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
	csv_path = os.path.join(os.path.dirname(__file__), "StudentPerformance.csv")
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Dataset not found at {csv_path}")
	df = pd.read_csv(csv_path)
	print("First 5 Rows:")
	print(df.head())
	print("\nDataset Info:")
	df.info()
	print("\nStatistical Summary:")
	print(df.describe())
	print("\nMissing Values:")
	print(df.isnull().sum())
	plt.figure()
	plt.hist(df["Performance Index"].dropna(), bins=10, edgecolor='black')
	plt.title("Performance Index Distribution")
	plt.xlabel("Performance Index")
	plt.ylabel("Frequency")
	plt.tight_layout()
	plt.show()
	activity_counts = df["Extracurricular Activities"].value_counts()
	plt.figure()
	plt.bar(activity_counts.index, activity_counts.values)
	plt.title("Extracurricular Activities Distribution")
	plt.xlabel("Extracurricular Activities")
	plt.ylabel("Count")
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	top_activities = df["Extracurricular Activities"].value_counts().nlargest(5).index
	scores_by_activity = [df[df["Extracurricular Activities"] == a]["Performance Index"].dropna() for a in top_activities]
	plt.figure()
	plt.boxplot(scores_by_activity, labels=top_activities)
	plt.title("Performance Index by Top Extracurricular Activities")
	plt.xlabel("Extracurricular Activities")
	plt.ylabel("Performance Index")
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	corr = df.corr(numeric_only=True)
	plt.figure()
	plt.imshow(corr, cmap='viridis', aspect='auto')
	plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
	plt.yticks(range(len(corr.columns)), corr.columns)
	plt.title("Correlation Heatmap")
	plt.colorbar()
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    main()