from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path: Path) -> pd.DataFrame:
    """Load and standardize the student exam score dataset."""
    print(f"[*] Loading dataset from '{file_path.name}'...")
    if not file_path.exists():
        print(f"[!] ERROR: Dataset file '{file_path}' was not found.")
        sys.exit(1)

    df = pd.read_csv(file_path, sep=r'\s+')
    df.columns = ['ID', 'MATH_SCORE', 'VERBAL_SCORE']
    return df


def calculate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics, distribution shape, and outliers."""
    print("\n" + "=" * 72)
    print("                 DESCRIPTIVE STATISTICAL ANALYSIS REPORT")
    print("=" * 72)

    subjects = ['MATH_SCORE', 'VERBAL_SCORE']
    summary_rows = []

    for subject in subjects:
        series = df[subject].dropna()
        n = len(series)
        mean_val = series.mean()
        median_val = series.median()
        mode_series = series.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan

        var_val = series.var(ddof=1)
        std_val = series.std(ddof=1)
        sem_val = std_val / np.sqrt(n)
        ci_95 = 1.96 * sem_val

        min_val, max_val = series.min(), series.max()
        q1, q2, q3 = np.percentile(series, [25, 50, 75])
        iqr_val = q3 - q1
        skew_val = series.skew()
        kurt_val = series.kurtosis()

        z_scores = np.abs((series - mean_val) / std_val)
        outliers = int(np.sum(z_scores > 2.5))

        label = subject.replace('_', ' ')
        print(f"\n[{label}] (n = {n})")
        print("-" * 45)
        print(f"  Mean ± 95% CI      : {mean_val:.2f} ± {ci_95:.2f}")
        print(f"  Median (Q2)        : {median_val:.2f}")
        print(f"  Mode               : {mode_val:.2f}")
        print(f"  Variance (s²)      : {var_val:.2f}")
        print(f"  Std. Deviation (s) : {std_val:.2f}")
        print(f"  Range (Min - Max)  : {min_val:.2f} - {max_val:.2f} (Span: {max_val - min_val:.2f})")
        print(f"  Quartiles          : Q1={q1:.2f} | Q2={q2:.2f} | Q3={q3:.2f} (IQR={iqr_val:.2f})")
        print(f"  Skewness / Kurtosis: {skew_val:.3f} / {kurt_val:.3f}")
        print(f"  Outliers (|Z|>2.5) : {outliers}")

        summary_rows.append({
            'Subject': label,
            'Count': n,
            'Mean': round(mean_val, 3),
            'Std_Dev': round(std_val, 3),
            'Variance': round(var_val, 3),
            'Min': round(min_val, 3),
            'Q1': round(q1, 3),
            'Median': round(median_val, 3),
            'Q3': round(q3, 3),
            'Max': round(max_val, 3),
            'IQR': round(iqr_val, 3),
            'Skewness': round(skew_val, 3),
            'Kurtosis': round(kurt_val, 3),
            'Outliers_Z2_5': outliers,
        })

    corr = df['MATH_SCORE'].corr(df['VERBAL_SCORE'])
    print("\n" + "-" * 72)
    print(f"  Pearson Correlation (Math vs. Verbal): r = {corr:.4f}")
    print("=" * 72)

    return pd.DataFrame(summary_rows)


def generate_visualizations(df: pd.DataFrame, output_path: Path) -> None:
    """Render a 2x2 analytical dashboard (histograms, boxplots, and regression scatter)."""
    print("\n[*] Generating statistical visualization dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Student Exam Performance — Statistical Distribution Dashboard', fontsize=14, fontweight='bold')

    # 1. Math Score Histogram
    ax0 = axes[0, 0]
    ax0.hist(df['MATH_SCORE'], bins=12, color='#2563eb', alpha=0.75, edgecolor='black')
    ax0.axvline(df['MATH_SCORE'].mean(), color='#dc2626', linestyle='--', linewidth=1.8, label=f"Mean ({df['MATH_SCORE'].mean():.1f})")
    ax0.axvline(df['MATH_SCORE'].median(), color='#f59e0b', linestyle='-', linewidth=1.8, label=f"Median ({df['MATH_SCORE'].median():.1f})")
    ax0.set_title('Math Score Distribution')
    ax0.set_xlabel('Score')
    ax0.set_ylabel('Frequency')
    ax0.legend()
    ax0.grid(alpha=0.25)

    # 2. Verbal Score Histogram
    ax1 = axes[0, 1]
    ax1.hist(df['VERBAL_SCORE'], bins=12, color='#059669', alpha=0.75, edgecolor='black')
    ax1.axvline(df['VERBAL_SCORE'].mean(), color='#dc2626', linestyle='--', linewidth=1.8, label=f"Mean ({df['VERBAL_SCORE'].mean():.1f})")
    ax1.axvline(df['VERBAL_SCORE'].median(), color='#f59e0b', linestyle='-', linewidth=1.8, label=f"Median ({df['VERBAL_SCORE'].median():.1f})")
    ax1.set_title('Verbal Score Distribution')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.25)

    # 3. Comparative Boxplot
    ax2 = axes[1, 0]
    bp = ax2.boxplot(
        [df['MATH_SCORE'], df['VERBAL_SCORE']],
        tick_labels=['Math Score', 'Verbal Score'],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(bp['boxes'], ['#93c5fd', '#6ee7b7']):
        patch.set_facecolor(color)
    ax2.set_title('Quartile & Outlier Comparison (Boxplot)')
    ax2.set_ylabel('Score')
    ax2.grid(axis='y', alpha=0.25)

    # 4. Scatter Plot + Linear Regression Fit
    ax3 = axes[1, 1]
    x = df['MATH_SCORE'].to_numpy()
    y = df['VERBAL_SCORE'].to_numpy()
    ax3.scatter(x, y, color='#4f46e5', alpha=0.65, edgecolors='none', label='Students')
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax3.plot(x_line, slope * x_line + intercept, color='#dc2626', linewidth=2, label=f'Fit: y = {slope:.2f}x + {intercept:.1f}')
    ax3.set_title('Math vs. Verbal Correlation & Linear Fit')
    ax3.set_xlabel('Math Score')
    ax3.set_ylabel('Verbal Score')
    ax3.legend()
    ax3.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[+] Dashboard saved to '{output_path.name}'.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / 'data' / 'scores.txt'
    if not dataset_path.exists():
        dataset_path = base_dir / 'data set.txt'

    student_data = load_data(dataset_path)
    summary_df = calculate_descriptive_statistics(student_data)
    generate_visualizations(student_data, base_dir / 'score_distribution.png')
