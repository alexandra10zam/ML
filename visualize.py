import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATA_PATH = "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTDIR = "figures"


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Ensure types
    if 'Diabetes_binary' in df.columns:
        df['Diabetes_binary'] = df['Diabetes_binary'].astype(int)

    # 1) Diabetes distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Diabetes_binary', data=df)
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.title('Diabetes distribution')
    plt.savefig(os.path.join(OUTDIR, 'diabetes_count.png'), bbox_inches='tight')
    plt.close()

    # 2) Correlation heatmap (numeric columns)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation matrix')
    plt.savefig(os.path.join(OUTDIR, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

    # 3) BMI by diabetes (boxplot)
    if 'BMI' in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x='Diabetes_binary', y='BMI', data=df)
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.title('BMI by Diabetes')
        plt.savefig(os.path.join(OUTDIR, 'bmi_by_diabetes.png'), bbox_inches='tight')
        plt.close()

    # 4) Pairplot of a small sample for quick relationships
    pair_cols = [c for c in ['BMI', 'Age', 'GenHlth', 'Diabetes_binary'] if c in df.columns]
    if len(pair_cols) >= 2:
        n = min(len(df), 1000)
        sns.pairplot(df[pair_cols].sample(n), hue='Diabetes_binary', diag_kind='hist', corner=True)
        plt.savefig(os.path.join(OUTDIR, 'pairplot_sample.png'), bbox_inches='tight')
        plt.close()

    # 5) Countplots for selected categorical/binary features
    cat_cols = [c for c in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump'] if c in df.columns]
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Counts: {col}')
        plt.savefig(os.path.join(OUTDIR, f'count_{col}.png'), bbox_inches='tight')
        plt.close()

    # 6) Save a small numeric summary
    df.describe().to_csv(os.path.join(OUTDIR, 'summary_stats.csv'))

    print(f"Saved figures and summary to: {OUTDIR}")


if __name__ == '__main__':
    main()
