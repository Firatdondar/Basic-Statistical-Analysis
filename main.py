import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    print(f"[*] Loading dataset from '{file_path}'...")
    try:
        df = pd.read_csv(file_path, sep=r'\s+')
        # Rename columns to standardized English formats
        df.columns = ['ID', 'MATH_SCORE', 'TURKISH_SCORE']
        return df
    except FileNotFoundError:
        print(f"[!] ERROR: The file '{file_path}' was not found.")
        exit(1)

def calculate_descriptive_statistics(df):
    print("\n--- DESCRIPTIVE STATISTICS ---")
    subjects = ['MATH_SCORE', 'TURKISH_SCORE']
    
    for subject in subjects:
        mean_val = df[subject].mean()
        median_val = df[subject].median()
        
        # Safe mode calculation
        mode_series = df[subject].mode()
        mode_val = mode_series[0] if not mode_series.empty else np.nan
        
        std_val = df[subject].std()
        q1, q2, q3 = np.percentile(df[subject], [25, 50, 75])
        
        print(f"\n[{subject.replace('_', ' ')}]")
        print(f"Mean               : {mean_val:.2f}")
        print(f"Median (Q2)        : {median_val:.2f}")
        print(f"Mode               : {mode_val}")
        print(f"Standard Deviation : {std_val:.2f}")
        print(f"Quartiles          : Q1: {q1:.2f} | Q2: {q2:.2f} | Q3: {q3:.2f}")

def generate_visualizations(df):
    print("\n[*] Generating visualizations...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Math Score Histogram
    ax[0].hist(df['MATH_SCORE'], bins=10, color='royalblue', alpha=0.7, edgecolor='black')
    ax[0].set_title('Math Score Distribution')
    ax[0].set_xlabel('Scores')
    ax[0].set_ylabel('Frequency')

    # Turkish Score Histogram
    ax[1].hist(df['TURKISH_SCORE'], bins=10, color='seagreen', alpha=0.7, edgecolor='black')
    ax[1].set_title('Turkish Score Distribution')
    ax[1].set_xlabel('Scores')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('score_distribution.png')
    print("[+] Visualizations successfully saved as 'score_distribution.png'.")

if __name__ == "__main__":
    dataset_path = 'data set.txt'
    
    student_data = load_data(dataset_path)
    calculate_descriptive_statistics(student_data)
    generate_visualizations(student_data)
