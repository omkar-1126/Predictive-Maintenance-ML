import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve
import pandas as pd
import os

os.makedirs('outputs', exist_ok=True)
sns.set_theme(style='whitegrid')


def plot_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Sensor Data - Exploratory Analysis', fontsize=16, fontweight='bold')

    features = ['temperature_C', 'vibration_mm_s', 'pressure_bar', 'rpm', 'oil_level_pct', 'equipment_age_days']
    for ax, feat in zip(axes.flatten(), features):
        sns.histplot(data=df, x=feat, hue='failure', kde=True, ax=ax, palette={0: 'steelblue', 1: 'tomato'})
        ax.set_title(feat.replace('_', ' ').title())

    plt.tight_layout()
    plt.savefig('outputs/eda_distributions.png', dpi=150)
    plt.close()
    print("Saved: outputs/eda_distributions.png")


def plot_correlation(df):
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png', dpi=150)
    plt.close()
    print("Saved: outputs/correlation_matrix.png")


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'], ax=ax)
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f} | AUC: {res['roc_auc']:.3f}")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=150)
    plt.close()
    print("Saved: outputs/confusion_matrices.png")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    colors = ['steelblue', 'tomato', 'seagreen']

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=150)
    plt.close()
    print("Saved: outputs/roc_curves.png")


def plot_feature_importance(importance_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150)
    plt.close()
    print("Saved: outputs/feature_importance.png")


def plot_dashboard(df, results, importance_df, y_test):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Predictive Maintenance — ML Dashboard', fontsize=18, fontweight='bold', y=1.01)

    ax1 = fig.add_subplot(3, 3, 1)
    colors = ['steelblue', 'tomato']
    counts = df['failure'].value_counts().sort_index()
    ax1.pie(counts, labels=['Normal', 'Failure'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Failure Distribution')

    ax2 = fig.add_subplot(3, 3, 2)
    df_str = df.copy()
    df_str['Status'] = df_str['failure'].map({0: 'Normal', 1: 'Failure'})
    sns.boxplot(data=df_str, x='Status', y='temperature_C', palette={'Normal': 'steelblue', 'Failure': 'tomato'}, hue='Status', legend=False, ax=ax2)
    ax2.set_title('Temperature vs Failure')

    ax3 = fig.add_subplot(3, 3, 3)
    sns.boxplot(data=df_str, x='Status', y='vibration_mm_s', palette={'Normal': 'steelblue', 'Failure': 'tomato'}, hue='Status', legend=False, ax=ax3)
    ax3.set_title('Vibration vs Failure')

    ax4 = fig.add_subplot(3, 3, 4)
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    aucs = [results[m]['roc_auc'] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    ax4.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    ax4.bar(x + width/2, aucs, width, label='ROC-AUC', color='tomato')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
    ax4.set_ylim(0.8, 1.0)
    ax4.set_title('Model Comparison')
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(3, 3, 5)
    sns.barplot(data=importance_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False, ax=ax5)
    ax5.set_title('Feature Importance (RF)')
    ax5.set_xlabel('')

    ax6 = fig.add_subplot(3, 3, 6)
    colors6 = ['steelblue', 'tomato', 'seagreen']
    for (name, res), color in zip(results.items(), colors6):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax6.plot(fpr, tpr, color=color, lw=2, label=f"{name[:6]}.. ({res['roc_auc']:.3f})")
    ax6.plot([0, 1], [0, 1], 'k--', lw=1)
    ax6.set_title('ROC Curves')
    ax6.legend(fontsize=7)

    ax7 = fig.add_subplot(3, 3, 7)
    sns.scatterplot(data=df.sample(500), x='equipment_age_days', y='temperature_C',
                    hue='failure', palette={0: 'steelblue', 1: 'tomato'}, alpha=0.6, ax=ax7)
    ax7.set_title('Age vs Temperature')
    ax7.legend(title='Failure', labels=['No', 'Yes'])

    ax8 = fig.add_subplot(3, 3, 8)
    sns.scatterplot(data=df.sample(500), x='vibration_mm_s', y='oil_level_pct',
                    hue='failure', palette={0: 'steelblue', 1: 'tomato'}, alpha=0.6, ax=ax8)
    ax8.set_title('Vibration vs Oil Level')
    ax8.legend(title='Failure', labels=['No', 'Yes'])

    ax9 = fig.add_subplot(3, 3, 9)
    best_cm = results['Random Forest']['confusion_matrix']
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'], ax=ax9)
    ax9.set_title('Best Model - Confusion Matrix\n(Random Forest)')

    plt.tight_layout()
    plt.savefig('outputs/dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/dashboard.png")
