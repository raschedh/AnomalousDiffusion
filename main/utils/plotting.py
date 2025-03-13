"""
This file contains various functions for generating publication-quality plots
with consistent scaling. The plots are primarily exported as SVGs for further 
editing in Inkscape. Many functions return SVG files. It's messy right now but see
plots.ipynb for usage. Needs to be tidied up and further commented. All a bunch of matplotlib stuff. 

"""

import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from andi_datasets.utils_challenge import *
from padding import LABEL_PADDING_VALUE
from postprocessing import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
import seaborn as sns
import os
import numpy as np
import pandas as pd 
from statsmodels.nonparametric.smoothers_lowess import lowess

K_COLOR = '#1B9E77'
ALPHA_COLOR = '#E69F00'
STATE_COLOR = '#9970AB'
ALPHA_K_FOCUSED_COLOR = 'black'
AKS_COLOR = 'black'
MODELS = ['imm.', 'conf.', 'free', 'dir.']
NO_OF_CPS_MAX = 5
TRACKS_PER_CATEGORY = 100
FONT_SIZE = 32
BAR_WIDTH = 0.25
LINE_WIDTH = 2
GREY_COLOR = '0.8'
NUM_CLASSES = 4
CAPSIZE = 8
MARKER_SIZE = 8

def plot_multiple_position_jaccard(d_positions_alpha, d_positions_k, d_positions_state, bin_size=20):
   # Create figure
   fig = Figure(figsize=(8, 6), dpi=300)
   canvas = FigureCanvasSVG(fig)
   ax = fig.add_subplot(111)
   
   def process_data(d_positions):
       results = {}
       for key, items in d_positions.items():
           if items:
               avg_jaccard = sum(items) / len(items)
               results[key] = {'average_jaccard': avg_jaccard}
       x_values = [int(key) + bin_size/2 for key in sorted(results.keys(), key=int)]
       y_values = [results[key]['average_jaccard'] for key in sorted(results.keys(), key=int)]
       return x_values, y_values
   
   datasets = [
       (d_positions_k, K_COLOR, "K"),
       (d_positions_alpha, ALPHA_COLOR, "α"),
       (d_positions_state, STATE_COLOR, "s")
   ]
   
   for data, color, label in datasets:
       x_values, y_values = process_data(data)
       ax.plot(x_values, y_values, 
               color=color,
               linewidth=2,
               alpha=1,
               marker='o',
               markersize=10,
               markerfacecolor=color,
               markeredgecolor=color,
               markeredgewidth=2,
               label=label)
   
   ax.set_ylim(0, 1.0)
   ax.set_yticks([0, 0.5, 1])
   ax.set_yticklabels(['0', '0.5', '1'])
   
   ax.set_xlim(0, 200)
   x_ticks = list(range(20, 201, 20))
   ax.set_xticks(x_ticks)
   ax.set_xticklabels(['20', '', '', '', '100', '', '', '', '', '200'])
   ax.set_axisbelow(True)
   
   ax.tick_params(which='both', direction='out', length=6, width=1,
                 colors='black', pad=2, labelsize=32)
   
   for spine in ax.spines.values():
       spine.set_visible(True)
       spine.set_linewidth(1)
       spine.set_color('black')
   
   ax.set_xlabel('Position', fontsize=32)
   ax.set_ylabel(r'$\overline{J}$', fontsize=32, rotation=0, va='center', labelpad=20)
   
   legend = ax.legend(fontsize=28,
                     frameon=True,
                     loc='upper right',
                     bbox_to_anchor=(0.98, 0.98),
                     edgecolor='none',
                     facecolor='white',
                     framealpha=0.8)
   
   fig.tight_layout()
   fig.patch.set_facecolor('none')
   ax.set_facecolor('none')
   
   canvas.print_figure('position_jaccard_comparison_all.svg',
                      bbox_inches='tight',
                      pad_inches=0.1,
                      format='svg')
   
   return fig, ax

def plot_jaccard_position(d_positions, label_type="alpha", bin_size=20):
    """Plot Jaccard values against changepoint positions with improved readability."""
    plt.close('all')
    plt.clf()
    
    # Set color based on label type
    if label_type == "alpha":
        COLOR = ALPHA_COLOR
    elif label_type == "k":
        COLOR = K_COLOR
    elif label_type == "state":
        COLOR = STATE_COLOR 
    else:
        COLOR = "black"
        
    # Calculate results
    results = {}
    for key, items in d_positions.items():
        if items:
            avg_jaccard = sum(items) / len(items)
            results[key] = {
                'average_jaccard': avg_jaccard,
                'num_tracks': len(items),
                'all_values': items
            }
    
    # Extract plotting values
    x_values = [int(key) + bin_size/2 for key in sorted(results.keys(), key=int)]
    y_values = [results[key]['average_jaccard'] for key in sorted(results.keys(), key=int)]
    
    # Save arrays
    np.save(f"jaccard_values_position_{label_type}_for_graph.npy", np.array(y_values))
    
    # Create figure
    fig = Figure(figsize=(10, 6), dpi=300)  # Increased width for better spacing
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    # Plot with both line and markers
    ax.plot(x_values, y_values, color=COLOR, 
            linewidth=2, alpha=1,
            marker='o', markersize=8,
            markerfacecolor='white',
            markeredgecolor=COLOR,
            markeredgewidth=2)
    
    # Set y-axis limits with margin
    ymin = min(y_values)
    ymax = max(y_values)
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)
    
    # Set x-axis ticks with improved spacing
    bin_edges = list(range(0, 201, bin_size))
    ax.set_xticks(bin_edges)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(bin_edges, rotation=45, ha='right')
    
    # Configure axis and style
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Configure tick parameters
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    ax.tick_params(axis='both', labelsize=32)
    
    # Configure spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # Set labels with adjusted positioning
    ax.set_xlabel('Position', fontsize=32, labelpad=20)  # Increased labelpad
    ax.set_ylabel(r'$\overline{J}$', fontsize=32, rotation=0, va='center', labelpad=20)
    
    # Adjust layout with more padding at bottom
    fig.tight_layout(pad=1.0, rect=[0, 0.1, 1, 1])  # Add bottom padding
    
    # Set transparent background
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Save as SVG
    canvas.print_figure(f'jaccard_vs_position_{label_type}.svg', 
                       bbox_inches='tight', 
                       pad_inches=0.1, 
                       format='svg')
    
    return


def plot_jaccard_cps(d_cps, label_type="alpha"): 
    # Print results
    plt.close('all')  # Close all figures
    plt.clf()         # Clear current figure
    n_cps = []  # List for number of changepoints
    jaccard_values = []
    plt.clf()

    if label_type == "alpha":
        COLOR = ALPHA_COLOR
    elif label_type == "k":
        COLOR = K_COLOR
    elif label_type == "state":
        COLOR = STATE_COLOR 
    else:
        COLOR = "black"
    # Calculate average Jaccard value for each number of changepoints
    results = {}
    for key, items in d_cps.items():
        avg_jaccard = sum(items) / len(items)
        num_tracks = len(items)
        results[key] = {
            'average_jaccard': avg_jaccard,
            'num_tracks': num_tracks,
            'all_values': items
        }
        
    print("\nResults by number of changepoints:")
    for key in sorted(results.keys()):
        print(f"\nNumber of changepoints: {key}")
        print(f"Average Jaccard value: {results[key]['average_jaccard']:.4f}")
        print(f"Number of tracks: {results[key]['num_tracks']}")

    # Create lists for plotting
    for key in sorted(results.keys()):
        n_cps.append(int(key))
        jaccard_values.append(results[key]['average_jaccard'])

    # Save arrays for future use
    # np.save("n_cps_for_graph.npy", np.array(n_cps))
    np.save("jaccard_values_"+str(label_type)+"_for_graph.npy", np.array(jaccard_values))

    # Create figure with specific DPI for precise control
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    # Plot with both line and markers
    ax.plot(n_cps, jaccard_values, color=COLOR, 
            linewidth=2, alpha=1,
            marker='o', markersize=8,
            markerfacecolor='white',  # White fill
            markeredgecolor=COLOR,  # Green border
            markeredgewidth=2)

    # Set y-axis limits with a small margin
    ymin = min(jaccard_values)
    ymax = max(jaccard_values)
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)

    # Set x-axis to show only integer values
    ax.set_xticks(n_cps)

    # Configure axis and style
    ax.set_axisbelow(True)

    # Configure tick parameters to match other plots
    ax.tick_params(which='both', direction='out', length=6, width=1,
                colors='black', pad=2)

    # Set tick label sizes
    ax.tick_params(axis='both', labelsize=32)

    # Configure spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # Set labels
    ax.set_xlabel('$N_{\mathrm{CP}}$', fontsize=32)
    ax.set_ylabel(r'$\overline{J}$', fontsize=32, rotation=0, va='center', labelpad=20)

    # Adjust layout
    fig.tight_layout(pad=1.0)

    # Set transparent background
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Save as SVG
    canvas.print_figure('jaccard_vs_changepoints_'+str(label_type)+'.svg', bbox_inches='tight', 
                    pad_inches=0.1, format='svg')

    return 


def plot_mae_cps(d_cps, label_type="alpha"):
   plt.close('all')
   plt.clf()
   n_cps = []  
   mae_values = []

   COLOR = ALPHA_COLOR if label_type == "alpha" else K_COLOR if label_type == "k" else STATE_COLOR if label_type == "state" else "black"

   # Calculate average MAE for each number of changepoints
   results = {}
   for key, items in d_cps.items():
       avg_mae = sum(items) / len(items)
       results[key] = {
           'average_mae': avg_mae,
           'num_tracks': len(items),
           'all_values': items
       }

   # Create lists for plotting
   for key in sorted(results.keys()):
       n_cps.append(int(key))
       mae_values.append(results[key]['average_mae'])

   # Save arrays
   np.save(f"mae_values_{label_type}_for_graph.npy", np.array(mae_values))

   # Create figure
   fig = Figure(figsize=(8, 6), dpi=300)
   canvas = FigureCanvasSVG(fig)
   ax = fig.add_subplot(111)

   # Plot
   ax.plot(n_cps, mae_values, color=COLOR,
           linewidth=2, alpha=1,
           marker='o', markersize=8,
           markerfacecolor=COLOR,
           markeredgecolor=COLOR,
           markeredgewidth=2)

   # Axis limits
    # Axis limits and ticks
   if label_type == "k":
       ymin, ymax = 0, 0.06
       ax.set_ylabel(r'MSLE(K)', fontsize=32, rotation=90, va='center', labelpad=20)
   else:
       ymin, ymax = min(mae_values), max(mae_values)
       ax.set_ylabel(r'MAE(\alpha)', fontsize=32, rotation=90, va='center', labelpad=20)

    
   margin = (ymax - ymin) * 0.1
   ax.set_ylim(ymin - margin, ymax + margin)
    
   ax.set_xticks(n_cps)
   # Style
   ax.set_axisbelow(True)
   ax.tick_params(which='both', direction='out', length=6, width=1,
                 colors='black', pad=2, labelsize=32)

   for spine in ax.spines.values():
       spine.set_visible(True)
       spine.set_linewidth(1)
       spine.set_color('black')

   # Labels
   ax.set_xlabel('$N_{\mathrm{CP}}$', fontsize=32)

   # Layout
   fig.tight_layout(pad=1.0)
   fig.patch.set_facecolor('none')
   ax.set_facecolor('none')

   # Save
   canvas.print_figure(f'plots/mae_vs_changepoints_{label_type}.svg',
                      bbox_inches='tight', pad_inches=0.1, format='svg')
   
   return 


def plot_mae_cps_combined(d_cps_k, d_cps_alpha):
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    for data, label_type in [(d_cps_k, "k"), (d_cps_alpha, "alpha")]:
        n_cps = []
        mae_values = []
        COLOR = K_COLOR if label_type == "k" else ALPHA_COLOR

        results = {}
        for key, items in data.items():
            avg_mae = sum(items) / len(items)
            results[key] = {
                'average_mae': avg_mae,
                'num_tracks': len(items),
                'all_values': items
            }

        for key in sorted(results.keys()):
            n_cps.append(int(key))
            mae_values.append(results[key]['average_mae'])

        np.save(f"mae_values_{label_type}_for_graph.npy", np.array(mae_values))

        ax.plot(n_cps, mae_values, color=COLOR,
               linewidth=2, alpha=1,
               marker='o', markersize=8,
               markerfacecolor=COLOR,
               markeredgecolor=COLOR,
               markeredgewidth=2,
               label=r'MALE(K)' if label_type == "k" else r'MAE($\alpha$)')

    ax.set_ylim(0, 0.14)
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    ax.set_xticks(n_cps)

    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.set_xlabel('$N_{\mathrm{CP}}$', fontsize=32)
    # ax.set_ylabel('MAE', fontsize=32, rotation=90, va='center', labelpad=20)
    
    ax.legend(fontsize=28, frameon=False)

    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    canvas.print_figure('plots/mae_vs_changepoints_combined_k_alpha_new.svg',
                       bbox_inches='tight', pad_inches=0.1, format='svg')
    
    return fig, ax


def plot_rmse_cps(d_cps, label_type="alpha"): 
    n_cps = []
    rmse_values = []

    if label_type == "alpha_without_0":
        COLOR = ALPHA_COLOR
    elif label_type == "k_without_0":
        COLOR = K_COLOR
    elif label_type == "state_without_0":
        COLOR = STATE_COLOR 
    else:
        COLOR = "black"

    results = {}
    for key, items in d_cps.items():
        valid_items = [x for x in items if not (np.isnan(x) or np.isinf(x))]   
        if valid_items:
            avg_rmse = sum(valid_items) / len(valid_items)
        else:
            avg_rmse = float('nan')  # or handle empty case differently

        results[key] = {
            'average_rmse': avg_rmse,
        }

    for key in sorted(results.keys()):
        n_cps.append(int(key))
        rmse_values.append(results[key]['average_rmse'])

    np.save("rmse_values_"+str(label_type)+"_for_graph.npy", np.array(rmse_values))

    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    ax.plot(n_cps, rmse_values, color=COLOR, 
            linewidth=2, alpha=1,
            marker='o', markersize=8,
            markerfacecolor='white',
            markeredgecolor=COLOR,
            markeredgewidth=2)

    ymin = min(rmse_values)
    ymax = max(rmse_values)
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xticks(n_cps)
    ax.tick_params(which='both', direction='out', length=6, width=1,
                colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.set_xlabel('$N_{\mathrm{CP}}$', fontsize=32)
    ax.set_ylabel('RMSE', fontsize=32, rotation=0, va='center', labelpad=20)

    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    canvas.print_figure('rmse_vs_changepoints_'+str(label_type)+'.svg', 
                    bbox_inches='tight', pad_inches=0.1, format='svg')

    return


def plot_model_jaccard(d_cps):
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    n_cps = list(range(NO_OF_CPS_MAX + 1))
    
    for model in MODELS:
        jaccard_values = [np.mean(cps) if cps else 0 for cps in d_cps[model]]
        line = ax.plot(n_cps, jaccard_values,
                color='black',
                linewidth=2,
                linestyle='-',
                marker='o',
                markersize=8,
                markerfacecolor='black',
                markeredgecolor='black',
                markeredgewidth=2)[0]
                
        # Add label at the end of the line
        end_x = n_cps[-1]
        end_y = jaccard_values[-1]
        ax.text(end_x + 0.1, end_y, model, fontsize=28, va='center')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.1, NO_OF_CPS_MAX + 1)  # Extended x-axis for labels
    ax.set_xticks(n_cps)
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.set_xlabel('$N_{\mathrm{CP}}$', fontsize=32)
    ax.set_ylabel(r'$\overline{J}$', fontsize=32, rotation=0, va='center', labelpad=20)
    
    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    canvas.print_figure('jaccard_vs_models.svg',
                       bbox_inches='tight', pad_inches=0.1, format='svg')
    
    return fig, ax



def plot_errors_by_model(mae_k, mae_a):
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    models = list(mae_k.keys())
    x_pos = np.arange(len(models))
    
    # Plot points
    k_scatter = ax.plot(x_pos, list(mae_k.values()),
            color=K_COLOR,
            marker='o',
            linewidth=2,
        #     linestyle = "--",
            label='MSLE(K)')
    
    a_scatter = ax.plot(x_pos, list(mae_a.values()),
            color=ALPHA_COLOR,
            marker='o',
            linewidth=2,
        #     linestyle = "--",
            label=r'MAE($\alpha$)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=32)
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

#     ax.set_xlim(-0.5, len(models) + 1)
    ax.set_ylim(-0.005, 0.16)  # Increased upper limit to fit legend
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    
    legend = ax.legend(fontsize=28, frameon=False, 
                    loc='upper center',
                    ncol=2,
                    labelcolor='black')

    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    canvas.print_figure('errors_by_model_new.svg',
                       bbox_inches='tight', pad_inches=0.1, format='svg')
    
    return fig, ax



def calculate_losses(gt_a, gt_k, gt_state, pred_a, pred_k, pred_state):
    """Calculate losses for each sequence length."""
    max_len = 181
    loss_a = np.zeros(max_len, dtype=np.float64)
    loss_k = np.zeros(max_len, dtype=np.float64)
    loss_state = np.zeros(max_len, dtype=np.float64)
    point_count = np.zeros(max_len, dtype=np.float64)

    # Calculate sequence lengths using padding mask (99)
    non_zero_mask = (gt_a != LABEL_PADDING_VALUE)
    sequence_lengths = non_zero_mask.sum(axis=1)
    indices = sequence_lengths - 20

    # Calculate masked differences
    diff_a = np.abs(pred_a - gt_a) * non_zero_mask
    diff_k = np.abs(pred_k - gt_k) * non_zero_mask
    diff_state = np.abs(pred_state - gt_state) * non_zero_mask

    # Sum for each sequence
    sums_a = diff_a.sum(axis=1)
    sums_k = diff_k.sum(axis=1)
    sums_state = diff_state.sum(axis=1)

    # Accumulate results and point counts
    np.add.at(loss_a, indices, sums_a)
    np.add.at(loss_k, indices, sums_k)
    np.add.at(loss_state, indices, sums_state)
    np.add.at(point_count, indices, sequence_lengths)

    # Calculate means
    valid_mask = point_count > 0
    loss_a[valid_mask] /= point_count[valid_mask]
    loss_k[valid_mask] /= point_count[valid_mask]
    loss_state[valid_mask] /= point_count[valid_mask]

    return loss_a, loss_k, loss_state

def create_loss_length_plot_smooth(gt_a, gt_k, gt_state, pred_a, pred_k, pred_state, output_dir):
    
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    loss_a, loss_k, loss_state = calculate_losses(gt_a, gt_k, gt_state, pred_a, pred_k, pred_state)
    
    x_values = np.arange(20, 201)
    colors = {'alpha': '#E69F00', 'K': '#1B9E77', 'state': '#9970AB'}
    
    # Apply LOWESS smoothing to each loss
    valid_points_a = ~np.isnan(loss_a)
    valid_points_k = ~np.isnan(loss_k)
    valid_points_s = ~np.isnan(loss_state)
    
    smoothed_a = lowess(loss_a[valid_points_a], x_values[valid_points_a], frac=0.1)
    smoothed_k = lowess(loss_k[valid_points_k], x_values[valid_points_k], frac=0.2)
    smoothed_s = lowess(loss_state[valid_points_s], x_values[valid_points_s], frac=0.3)
    
    # Plot smoothed lines
    ax.plot(smoothed_a[:, 0], smoothed_a[:, 1], color=colors['alpha'], linewidth=4, alpha=1, label=r'$\mathcal{L}_{\alpha}$')
    ax.plot(smoothed_k[:, 0], smoothed_k[:, 1], color=colors['K'], linewidth=4, alpha=1, label=r'$\mathcal{L}_{K}$')
    ax.plot(smoothed_s[:, 0], smoothed_s[:, 1], color=colors['state'], linewidth=4, alpha=1, label=r'$\mathcal{L}_{s}$')
    
    ax.set_xlim(15, 205)
    
    # Set specific x and y ticks
    x_ticks = list(range(20, 201, 20))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['20', '', '', '', '100', '', '', '', '', '200'])
    
    # Configure tick parameters and axes
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    ax.tick_params(axis='both', labelsize=32)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # Set labels with adjusted positioning
    ax.set_xlabel('Traj. Length', fontsize=32, labelpad=20)
    ax.set_ylabel('Loss', fontsize=32, labelpad=20)
    
    # Configure legend
    ax.legend(fontsize=28, frameon=False, 
             loc='upper right',
             handletextpad=0.5,
             columnspacing=0.6,
             markerscale=2.0)
    
    fig.tight_layout(pad=1.0)
    
    output_path = os.path.join(output_dir, 'loss_per_length_smooth_new.svg')
    canvas.print_figure(output_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    
    return fig, ax


def create_heatmap(true_values, pred_values, error, param_name, value_range, output_dir, color):
    # Calculate error
    if param_name.lower() == 'alpha':
        error_text = f'MAE = {error:.3f}'
    else:
        error_text = f'MRE = {error:.3f}'

    mask = true_values != LABEL_PADDING_VALUE
    true_values = true_values[mask]
    pred_values = pred_values[mask]

    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    # Calculate histogram
    num_bins = 40
    min_val, max_val = value_range
    true_bins = np.linspace(min_val, np.max(true_values), num_bins + 1)
    pred_bins = np.linspace(min_val, np.max(true_values), num_bins + 1)
    hist, xedges, yedges = np.histogram2d(true_values, pred_values, 
                                         bins=[true_bins, pred_bins])
    
    # Create colormap
    cmap_dict = {
        "alpha_color": ['white', ALPHA_COLOR],
        "state_color": ['white', STATE_COLOR],
        "k_color": ['white', K_COLOR]
    }
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', cmap_dict[color.lower()])
    
    # Plot heatmap
    im = ax.imshow(hist.T, cmap=cmap, aspect='equal',
                   extent=[min_val, max_val, min_val, max_val], origin='lower')

    # Add error text
    ax.text(0.05, 0.95, error_text,
            transform=ax.transAxes,
            fontsize=32,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=5))
    
    # Set ticks
    mid_val = (min_val + max_val) / 2
    tick_positions = [min_val, mid_val, max_val]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    tick_labels = [f'{int(x)}' if float(x).is_integer() else f'{x:.1f}' for x in tick_positions]
    ax.set_xticklabels(tick_labels, fontsize=32)
    ax.set_yticklabels(tick_labels, fontsize=32)

    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # Set labels
    # Set labels based on parameter with smaller font size
    if param_name.lower() == 'alpha':
        xlabel = r'$\alpha_{\mathrm{pred}}$'
        ylabel = r'$\alpha_{\mathrm{true}}$'
        filename = "alpha_heatmap_new.svg"
    else:  # K
        xlabel = r'$K_{\mathrm{pred}}$'
        ylabel = r'$K_{\mathrm{true}}$'
        filename = "K_heatmap_new.svg"
    
    
    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    
    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    output_path = os.path.join(output_dir, filename)
    canvas.print_figure(output_path, bbox_inches='tight', 
                       pad_inches=0.1, format='svg')

    return fig, ax



def plot_confusion_matrix(cm_normalized, output_dir):
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    cm_normalized = cm_normalized * 100  # Convert to percentages
    
    labels = ['imm.', 'conf.', 'free', 'dir.']
    cmap = mcolors.LinearSegmentedColormap.from_list('white_purple', 
                                                    ['white', STATE_COLOR])
    
    sns.heatmap(cm_normalized, 
                annot=True,
                fmt='.0f',
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels,
                square=True,
                annot_kws={'size': 32},
                cbar=False,
                linewidths=1,
                linecolor='black',
                robust=True,
                ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=32, labelpad=20)
    ax.set_ylabel('Ground Truth', fontsize=32, labelpad=20)
    
    ax.tick_params(labelsize=32, which='both', direction='out', 
                  length=6, width=1, colors='black', pad=2)
    
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=cm_normalized.shape[0], color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=cm_normalized.shape[1], color='black', linewidth=1)
    
    fig.tight_layout(pad=1.0)
    
    canvas.print_figure(os.path.join(output_dir, "confusion_matrix.svg"),
                       bbox_inches='tight', pad_inches=0.1, format='svg')
    
    return fig, ax


def calculate_normalized_confusion_matrix(pred_states, gt_states, num_classes=4):
    # Flatten the arrays and remove padding
    pred_flat = pred_states.flatten()
    gt_flat = gt_states.flatten()
    mask = (gt_flat != LABEL_PADDING_VALUE)
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    # Calculate confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    # Normalize the confusion matrix by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm_normalized


def calculate_metrics(pred_states, gt_states, num_classes=4):
    """
    Calculate F1 Score, Recall, Precision and average class accuracy
    """
    # Flatten and remove padding
    pred_flat = pred_states.flatten()
    gt_flat = gt_states.flatten()
    mask = (gt_flat != LABEL_PADDING_VALUE)
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(gt_flat, pred_flat, 
                                                                   labels=range(num_classes), 
                                                                   zero_division=0)
    
    # Calculate per-class accuracy from confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    avg_class_accuracy = np.mean(class_accuracy)
    
    # Create a formatted output string
    labels = ['imm.', 'comp.', 'free', 'dir.']
    
    print("\nDetailed Classification Metrics:")
    print("--------------------------------")
    for i, label in enumerate(labels):
        print(f"\n{label}:")
        print(f"F1 Score:   {f1[i]:.3f}")
        print(f"Precision:  {precision[i]:.3f}")
        print(f"Recall:     {recall[i]:.3f}")
        print(f"Accuracy:   {class_accuracy[i]:.3f}")
    
    print("\nAverages:")
    print("--------------------------------")
    print(f"Average F1 Score:           {np.mean(f1):.3f}")
    print(f"Average Precision:          {np.mean(precision):.3f}")
    print(f"Average Recall:             {np.mean(recall):.3f}")
    print(f"Average Class Accuracy:     {avg_class_accuracy:.3f}")
    
    return {
        'per_class': {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': class_accuracy
        },
        'averages': {
            'f1': np.mean(f1),
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'accuracy': avg_class_accuracy
        }
    }




def create_jaccard_difference_plot(file_path, counter_path, parameter_type='alpha'):
    """
    Create publication quality plot for Jaccard index vs either delta alpha or delta K
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file containing the Jaccard index data
    counter_path : str
        Path to the JSON file containing the counter data
    parameter_type : str
        Either 'alpha' or 'K' to specify which parameter is being plotted
    output_dir : str or None
        Directory to save the plot. If None, saves in current directory
    
    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects
    """
        
    if parameter_type not in ['alpha', 'K']:
        raise ValueError("parameter_type must be either 'alpha' or 'K'")

    if parameter_type == 'alpha':
        bin_width = 0.1
        x_limits = (-2, 2)
        xlabel = r'$\Delta\alpha$'
        output_filename = 'jaccard_delta_alpha_new.svg'
        color_variable = ALPHA_COLOR
        x_tick_positions = [-2, -1, 0, 1, 2]
    else:
    #    bin_width = 0.1
    #    x_limits = (-5, 5)
        x_limits = (-0.5, 0.5)
        xlabel = r'$\Delta K$' 
        output_filename = 'jaccard_delta_k_new_log_space.svg'
        color_variable = K_COLOR
        x_tick_positions = [-0.5, -0.4, -0.3, -0.25,-0.1, 0, 0.1, 0.25, 0.3, 0.4, 0.5]
        # x_tick_positions = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

    with open(file_path, 'r') as file:
        data = json.load(file)
    with open(counter_path, 'r') as file:
        counter_data = json.load(file)

    # Create a sorted list of keys to ensure same order
    keys = sorted(data.keys(), key=float)  # Sort keys as float numbers
    # Create matched arrays
    x_values = np.array([float(key) for key in keys])
    y_values = np.array([data[key] for key in keys])
    counter = np.array([counter_data[key] for key in keys])

    # Perform the division
    y_values = y_values / counter
    sorted_x = x_values
    sorted_y = y_values

    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    ax.plot(sorted_x, sorted_y,
            color=color_variable,
            linewidth=2, 
            alpha=1,
            marker='o',
            markersize=10,
            markerfacecolor=color_variable,
            markeredgecolor=color_variable,
            markeredgewidth=2)

    ax.set_xlim(x_limits)
    ax.set_ylim(0, 1.05)

    ax.set_xticks(x_tick_positions)
    ax.set_yticks([0, 0.5, 1])

    ax.set_axisbelow(True)

    ax.tick_params(which='both', direction='out', length=6, width=1,
                    colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel('$\overline{J}$', fontsize=32, rotation=0, va='center', labelpad=20)

    fig.tight_layout()
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    canvas.print_figure(output_filename,
                        bbox_inches='tight',
                        pad_inches=0,
                        format='svg')

    return fig, ax



def plot_prediction(pred, true, label_type, save_path=None):
    K_COLOR = '#1B9E77'      # Green
    ALPHA_COLOR = '#E69F00'  # Orange
    STATE_COLOR = '#9970AB'  # Purple
    
    if label_type =="alpha":
        COLOR = ALPHA_COLOR 
        ymin = 0
        ymax = 2
        label = r'$K$'
    elif label_type == "k":
        COLOR = K_COLOR
        ymin = 0
        ymax = 3
        label = r'$α$'
    elif label_type == "state":
        COLOR = STATE_COLOR
        ymin = 0
        ymax = 4
        label = r'$s$'
    else:
        raise ValueError

    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    time = np.arange(len(pred))
    ax.plot(time, pred, color=COLOR, linestyle='none', 
            marker='o', markersize=6, markerfacecolor='none',
            markeredgecolor='#1B9E77', markeredgewidth=1.5,
            alpha=0.7)
    
    cp_gt = get_cps_gt(true)

    for i in range(len(cp_gt)-1):
        start_idx = cp_gt[i]
        end_idx = cp_gt[i+1]
        if end_idx == len(true):
            end_idx -= 1
        segment_value = true[start_idx]  # value for this segment
        ax.plot([start_idx, end_idx], [segment_value, segment_value],
                color='black', linewidth=2, alpha=1)

    margin = (ymax - ymin) * 0.1  # 10% margin
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_axisbelow(True)
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    
    ax.tick_params(axis='both', labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # Set labels with exact same formatting as heatmap
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel(label, fontsize=32)
    
    # # Configure legend with matching font size
    # ax.legend(fontsize=28, frameon=True, loc='best')
    fig.tight_layout(pad=1.0)
    
    # Save as SVG with same parameters
    if save_path:
        canvas.print_figure(save_path, bbox_inches='tight', 
                          pad_inches=0.1, format='svg')
    
    return fig, ax



def create_plots_for_track_examples(example_track, pred_a, pred_k, pred_state):

    # Plot track 
    plt.figure(figsize=(10, 6))
    plt.plot(example_track["x"], example_track["y"], marker='o')    
    plt.figure()
    
    # Find where padding starts
    padding_starts = (example_track["alpha"] == LABEL_PADDING_VALUE).argmax() 
    if padding_starts == 0:
        padding_starts = 200

    # Trim predictions to remove padding
    pred_a = pred_a[:padding_starts]
    pred_k = pred_k[:padding_starts]
    pred_state = pred_state[:padding_starts]

    # Apply smoothing and value constraints
    pred_a = np.clip(median_filter_1d(smooth_series(pred_a)), 0, 2)
    pred_k = np.clip(median_filter_1d(smooth_series(pred_k)), 0, 6)
    pred_state = replace_short_sequences(pred_state)

    # Get ground truth values
    gt_alpha = example_track["alpha"][:padding_starts]
    gt_k = np.log10(example_track["D"][:padding_starts] + 1)
    gt_state = example_track["state"][:padding_starts]
    
    # Time points for x-axis
    time_points = np.arange(len(pred_a))

    # Plot Alpha
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, gt_alpha, linewidth=2)
    plt.scatter(time_points, pred_a, color=ALPHA_COLOR, alpha=0.6, s=30)
    plt.title("Alpha")
    plt.ylim(0, 2)
    plt.figure()

    # Plot K
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, gt_k, linewidth=2)
    plt.scatter(time_points, pred_k, color=K_COLOR, alpha=0.6, s=30)
    plt.title("K")
    plt.ylim(0, 3)
    plt.figure()

    # Plot State
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, gt_state, linewidth=2)
    plt.scatter(time_points, pred_state, color=STATE_COLOR, alpha=0.6, s=30)
    plt.title("State")
    plt.ylim(0, 4)
    plt.show()
    return 



def plot_tensorboard_metrics(train_csv, val_csv, save_path=None, loss_type='alpha'):
    """
    Plot training metrics from TensorBoard CSV files with consistent publication styling
    
    Parameters:
    -----------
    train_csv : str
        Path to training loss CSV file
    val_csv : str
        Path to validation loss CSV file
    save_path : str or None
        Path to save the output plot
    loss_type : str
        Either 'alpha' or 'K' to specify which loss is being plotted
    """
    # Read CSV files
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Extract step and value columns
    train_steps = train_df['Step']
    train_values = train_df['Value']
    val_steps = val_df['Step']
    val_values = val_df['Value']

    # Create figure with same dimensions as jaccard plot
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    # Plot data with solid and dashed lines in black
    ax.plot(train_steps, train_values, color='black', linewidth=3, 
            label='Train', linestyle='-')
    ax.plot(val_steps, val_values, color='black', linewidth=3, 
            label='Val', linestyle='--')

    # Configure axis and style to match jaccard plot
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(max(train_steps), max(val_steps)))

    # Configure tick parameters to match
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    
    # Set tick label sizes to match (32 pt)
    ax.tick_params(axis='both', labelsize=32)

    # Configure spines to match
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # Set labels with matching font sizes (55 pt)
    ax.set_xlabel('Epoch', fontsize=55)
    
    # Set y-label based on loss type using LaTeX notation
    if loss_type.lower() == 'alpha':
        ylabel = r'$L_{\alpha}$'
    else:  # K
        ylabel = r'$L_{K}$'
    ax.set_ylabel(ylabel, fontsize=55)
    
    # Configure legend with larger font size
    ax.legend(fontsize=32, frameon=True, loc='upper right')

    # Adjust layout
    fig.tight_layout(pad=1.0)

    # Save the plot
    if save_path:
        canvas.print_figure(save_path, bbox_inches='tight', 
                          pad_inches=0.1, format='svg')
    
    plt.show()
    
    return fig, ax



def plot_track_variable(pred, true, label_type, save_path=None):
    plot_params = {
        "alpha": {
            "color": ALPHA_COLOR,
            "ymin": 0,
            "ymax": 2,
            "label": r'$α$',
            "yticks": [0, 2]
        },
        "k": {
            "color": K_COLOR,
            "ymin": 0,
            "ymax": 1,
            "label": r'$D$',
            "yticks": [0, 1]
        },
        "state": {
            "color": STATE_COLOR,
            "ymin": 0,
            "ymax": 3,
            "label": r'$s$',
            "yticks": [0, 1, 2, 3]
        }
    }
    
    if label_type not in plot_params:
        raise ValueError(f"label_type must be one of {list(plot_params.keys())}")
    
    params = plot_params[label_type]

    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    # Plot ground truth segments
    cp_gt = get_cps_gt(true)
    for i in range(len(cp_gt)-1):
        start_idx = cp_gt[i]
        end_idx = cp_gt[i+1]
        if end_idx == len(true):
            end_idx -= 1
        segment_value = true[start_idx]
        
        # Plot predictions for each segment
        segment_length = end_idx - start_idx
        if segment_length <= 6:  # For short segments
            segment_time = np.linspace(start_idx, end_idx, 3)
        else:
            segment_time = np.arange(start_idx, end_idx, 12)  # Take every 3rd point
            if end_idx not in segment_time:  # Include end point
                segment_time = np.append(segment_time, end_idx)
            
        segment_pred = np.interp(segment_time, np.arange(len(pred)), pred)
        
        ax.plot(segment_time, segment_pred, color=params["color"], linestyle='none',
                marker='o', markersize=30, markerfacecolor=params["color"],
                markeredgecolor=params["color"], markeredgewidth=3,
                alpha=1)
        
        ax.plot([start_idx, end_idx], [segment_value, segment_value],
                color='black', linewidth=6, alpha=1)
        
    margin = (params["ymax"] - params["ymin"]) * 0.05
    ax.set_ylim(params["ymin"] - margin, params["ymax"] + margin)
    ax.set_axisbelow(True)
    ax.set_yticks(params["yticks"])
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2, labelsize=32)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel(params["label"], fontsize=32)
    ax.set_xticks([0, len(pred)])
    
    fig.tight_layout(pad=1.0)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    if save_path:
        canvas.print_figure(save_path, bbox_inches='tight',
                          pad_inches=0.1, format='svg')
    
    return fig, ax



def plot_colored_trajectory(df, save_path=None, multi=False):
    """
    Plot trajectory with different colors based on either:
    - State values (default): Blue for state 2, Red for states 0/1
    - Alpha changes (if multi=True): Alternates between blue and red when alpha changes
    """
    # Create figure
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)
    
    # Create masks for different states
    state_2_mask = df['state'] == 2
    state_01_mask = df['state'].isin([0, 1])
    
    # # Plot points for each state (commented out)
    # ax.scatter(df.loc[state_2_mask, 'x'], 
    #           df.loc[state_2_mask, 'y'],
    #           color='#362cfc',
    #           s=50,
    #           alpha=1,
    #           label='State 2')
              
    # ax.scatter(df.loc[state_01_mask, 'x'], 
    #           df.loc[state_01_mask, 'y'],
    #           color='#ff0000',
    #           s=50,
    #           alpha=1,
    #           label='State 0/1')
    
    if multi:
        # For multi mode, detect alpha changes and alternate colors
        use_blue = True  # Start with blue
        current_alpha = df['alpha'].iloc[0]
        
        for i in range(len(df)-1):
            # Check if alpha changed
            if df['alpha'].iloc[i+1] != current_alpha:
                use_blue = not use_blue  # Switch color
                current_alpha = df['alpha'].iloc[i+1]
            
            color = '#362cfc' if use_blue else '#ff0000'
            ax.plot([df['x'].iloc[i], df['x'].iloc[i+1]], 
                    [df['y'].iloc[i], df['y'].iloc[i+1]], 
                    color=color,
                    linewidth=1)
    else:
        # Original state-based coloring
        for i in range(len(df)-1):
            color = '#362cfc' if df['state'].iloc[i+1] == 2 else '#ff0000'
            ax.plot([df['x'].iloc[i], df['x'].iloc[i+1]], 
                    [df['y'].iloc[i], df['y'].iloc[i+1]], 
                    color=color,
                    linewidth=1)
    
    # Style the plot
    ax.set_xlabel('X', fontsize=32)
    ax.set_ylabel('Y', fontsize=32)
    
    # Configure ticks
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2)
    ax.tick_params(axis='both', labelsize=32)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    ax.set_axisbelow(True)
    
    # Add legend with matching font size (commented out since scatter is disabled)
    # ax.legend(fontsize=28, frameon=True, loc='best')
    
    # Adjust layout
    fig.tight_layout(pad=1.0)
    
    # Save if path provided
    if save_path:
        canvas.print_figure(save_path, bbox_inches='tight',
                          pad_inches=0.1, format='svg')
    
    return fig, ax


def plot_time_series_prediction(time, pred, truth, variable_type, save_path=None):
    """
    Create publication quality plot comparing predictions and ground truth
    
    Parameters:
    -----------
    time : array-like
        Time points
    pred : array-like
        Prediction values
    truth : array-like
        Ground truth values
    variable_type : str
        One of 'alpha', 'K', or 'state' to determine color scheme
    save_path : str, optional
        Path to save the output plot
    """
    # Set color based on variable type
    if variable_type.lower() == 'alpha':
        color = ALPHA_COLOR 
    elif variable_type.lower() == 'k':
        color = K_COLOR
    else:  # state
        color = STATE_COLOR
    
    # Create figure
    fig = Figure(figsize=(8, 6), dpi=300)
    canvas = FigureCanvasSVG(fig)
    ax = fig.add_subplot(111)

    # Plot predictions with markers
    ax.plot(time, pred, color=color, linestyle='none', 
            marker='o', markersize=6, alpha=0.7,
            label='Prediction')
            
    # Plot ground truth with solid line
    ax.plot(time, truth, color=color, linestyle='-', 
            linewidth=2, alpha=1,
            label='Ground Truth')

    # Configure axis and style
    ax.set_axisbelow(True)
    
    # Configure tick parameters
    ax.tick_params(which='both', direction='out', length=6, width=1,
                  colors='black', pad=2, labelsize=32)

    # Configure spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # Set labels
    ax.set_xlabel('Time', fontsize=55)
    
    # Set y-label based on variable type
    if variable_type.lower() == 'alpha':
        ylabel = r'$\alpha$'
    elif variable_type.lower() == 'k':
        ylabel = r'$K$'
    else:  # state
        ylabel = 'State'
    ax.set_ylabel(ylabel, fontsize=55)
    
    # Configure legend
    ax.legend(fontsize=32, frameon=True, loc='best')

    # Adjust layout
    fig.tight_layout(pad=1.0)

    # Save the plot
    if save_path:
        canvas.print_figure(save_path, bbox_inches='tight', 
                          pad_inches=0.1, format='svg')
    
    plt.show()
    
    return fig, ax