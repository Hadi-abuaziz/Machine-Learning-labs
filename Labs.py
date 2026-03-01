import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")


class DataLoader:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def check_dtypes(self):
        print(self.df.dtypes)
        print(self.df.describe())

    def encode(self, column):
        self.df[column] = self.df[column].map({"Yes": 1, "No": 0})
        return self.df


class MissingValues:
    def __init__(self, df):
        self.df = df.copy()

    def detect(self):
        print(self.df.isna().sum())

    def introduce(self, column, n=10):
        df_missing = self.df.copy()
        df_missing.loc[0:n-1, column] = np.nan
        return df_missing

    def remove(self, df_missing):
        return df_missing.dropna()

    def mean_impute(self, df_missing, column):
        df = df_missing.copy()
        df[column] = df[column].fillna(df[column].mean())
        return df

    def median_impute(self, df_missing, column):
        df = df_missing.copy()
        df[column] = df[column].fillna(df[column].median())
        return df


class Outliers:
    def __init__(self, df):
        self.df = df.copy()

    def boxplot(self, columns):
        fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
        for ax, col in zip(axes, columns):
            sns.boxplot(y=self.df[col], ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        plt.show()

    def detect_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        print(f"Lower: {lower:.2f}  Upper: {upper:.2f}  Outliers: {len(outliers)}")
        return lower, upper

    def remove(self, column):
        lower, upper = self.detect_iqr(column)
        return self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]

    def cap(self, column, low=0.05, high=0.95):
        df = self.df.copy()
        df[column] = df[column].clip(df[column].quantile(low), df[column].quantile(high))
        return df


class Normalizer:
    def __init__(self, df, columns):
        self.df = df[columns].copy()
        self.columns = columns

    def minmax(self):
        df = self.df.copy()
        df[self.columns] = MinMaxScaler().fit_transform(df)
        print(df.head())
        return df

    def zscore(self):
        df = self.df.copy()
        df[self.columns] = StandardScaler().fit_transform(df)
        print(df.head())
        return df


class PCAReducer:
    def __init__(self, df, columns):
        self.df = df[columns].copy()
        self.columns = columns

    def correlation_heatmap(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def apply(self, n_components=None):
        n = n_components or len(self.columns)
        pca = PCA(n_components=n)
        components = pca.fit_transform(self.df)
        print("Explained Variance:", pca.explained_variance_ratio_)
        print("Cumulative        :", pca.explained_variance_ratio_.cumsum())

        plt.figure(figsize=(7, 4))
        plt.bar(range(1, n+1), pca.explained_variance_ratio_, label="Individual")
        plt.plot(range(1, n+1), pca.explained_variance_ratio_.cumsum(), "o-r", label="Cumulative")
        plt.xlabel("Component")
        plt.ylabel("Variance Ratio")
        plt.title("Scree Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.scatter(components[:, 0], components[:, 1], alpha=0.3, s=5)
        plt.title("PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

        return components


FILEPATH = "StudentPerformance.csv"
TARGET = "Performance Index"
NUM_COLS = ["Hours Studied", "Previous Scores", "Sleep Hours",
            "Sample Question Papers Practiced", "Performance Index"]

loader = DataLoader(FILEPATH)
loader.check_dtypes()
df = loader.encode("Extracurricular Activities")

mv = MissingValues(df)
mv.detect()
df_missing = mv.introduce(TARGET)
mv.remove(df_missing)
mv.mean_impute(df_missing, TARGET)
mv.median_impute(df_missing, TARGET)

out = Outliers(df)
out.boxplot(NUM_COLS)
out.detect_iqr(TARGET)
out.remove(TARGET)
df = out.cap(TARGET)

norm = Normalizer(df, NUM_COLS)
norm.minmax()
df_std = norm.zscore()

pca = PCAReducer(df_std, NUM_COLS)
pca.correlation_heatmap()
pca.apply(n_components=5)