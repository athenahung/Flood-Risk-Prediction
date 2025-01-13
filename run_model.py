import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lstm import lstm
from itertools import product

results = []

def calculate_f1(conf_matrix):
    tn, fp = conf_matrix[0]
    fn, tp = conf_matrix[1]

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0

    return f1

def run_lstm_experiments(numTrials, time_steps, max_depths, random_state):
    for time_step, max_depth in product(time_steps, max_depths):
        losses = np.zeros(numTrials)
        accuracies = np.zeros(numTrials)
        f1_scores = np.zeros(numTrials)

        for i in range(numTrials):
            loss_i, acc_i, cm_i = lstm(time_step, max_depth, random_state)

            losses[i] = loss_i
            accuracies[i] = acc_i
            f1_scores[i] = calculate_f1(cm_i)
        
        mean_loss = np.mean(losses)
        mean_acc = np.mean(accuracies)
        mean_f1 = np.mean(f1_scores)

        std_loss = np.std(losses)
        std_acc = np.std(accuracies)
        std_f1 = np.std(f1_scores)

        results.append({
            'time_step': time_step,
            'max_depth': max_depth,
            'mean_loss': mean_loss,
            'mean_acc': mean_acc,
            'mean_f1': mean_f1,
            'std_loss': std_loss,
            'std_acc': std_acc,
            'std_f1': std_f1
        })

    return pd.DataFrame(results)

def plot_results(results):
    #plt.style.use('seaborn')
    sns.set_theme()
    fig = plt.figure(figsize=(15, 10))

    #1. F1 Score Heatmap
    ax1 = plt.subplot(221)
    pivot_f1 = results.pivot(index='time_step', columns='max_depth', values='mean_f1')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Mean F1 Score')
    ax1.set_xlabel('Max_Depth')
    ax1.set_ylabel('Time Steps')

    #2. Loss Heatmap
    ax2 = plt.subplot(222)
    pivot_loss = results.pivot(index='time_step', columns='max_depth', values='mean_loss')
    sns.heatmap(pivot_loss, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=ax2)
    ax2.set_title('Mean Loss')
    ax2.set_xlabel('Max Depth')
    ax2.set_ylabel('Time Steps')

    #3. F1 Score vs time_steps
    ax3 = plt.subplot(223)
    for depth in results['max_depth'].unique():
        mask = results['max_depth'] == depth
        plt.plot(results[mask]['time_step'], 
                results[mask]['mean_f1'], 
                'o-', 
                label=f'Depth={depth}')
        # Add error bars
        plt.fill_between(results[mask]['time_step'],
                        results[mask]['mean_f1'] - results[mask]['std_f1'],
                        results[mask]['mean_f1'] + results[mask]['std_f1'],
                        alpha=0.2)
    ax3.set_title('Time Steps vs F1 Score for Different Depths')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('F1 Score')
    ax3.legend()

    #4. Best Models
    ax4 = plt.subplot(224)
    ax4.axis('off')
    best_f1_idx = results['mean_f1'].idxmax()
    best_loss_idx = results['mean_loss'].idxmin()
    
    table_text = [
        ['Metric', 'Time Step', 'Max Depth', 'F1 Value', 'Std Dev'],
        ['Best F1', 
         f"{results.iloc[best_f1_idx]['time_step']}", 
         f"{results.iloc[best_f1_idx]['max_depth']}", 
         f"{results.iloc[best_f1_idx]['mean_f1']:.3f}",
         f"{results.iloc[best_f1_idx]['std_f1']:.3f}"],
        ['Best Loss', 
         f"{results.iloc[best_loss_idx]['time_step']}", 
         f"{results.iloc[best_loss_idx]['max_depth']}", 
         f"{results.iloc[best_loss_idx]['mean_loss']:.3f}",
         f"{results.iloc[best_loss_idx]['std_loss']:.3f}"]
    ]

    table = ax4.table(cellText=table_text, 
                     loc='center', 
                     cellLoc='center', 
                     bbox=[0, 0.3, 1, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Best Models')

    plt.tight_layout()
    return fig

time_steps = [5, 10, 15, 20]
max_depths = [1, 2, 3, 4, 5]
# time_steps = [5, 10]
# max_depths = [1, 2]
random_state = 42
numTrials = 10

results = run_lstm_experiments(numTrials, time_steps, max_depths, random_state)
fig = plot_results(results)
plt.show()

print('Detailed Results:')
print(results.to_string(index=False))