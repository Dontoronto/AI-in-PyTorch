import argparse
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Generate a bar plot from an Excel file.')
    parser.add_argument('--path', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--sheet', type=str, required=True, help='Sheet name in the Excel file')
    parser.add_argument('--filter', type=str, required=False, help='Filter string for column names')
    parser.add_argument('--header', type=int, default=1, help='Row number for the header')
    parser.add_argument('--bar_width', type=float, required=True, help='Width of the bars')
    args = parser.parse_args()

    offset = 1
    rows_head = args.header
    headers = list(np.arange(rows_head))

    # Load the Excel file
    df = pd.read_excel(args.path, sheet_name=args.sheet, header=headers)

    df.columns = [' '.join(col).strip() for col in df.columns.values]

    if args.filter is not None:
        labels = [col for col in df.columns if args.filter in col]
    else:
        print(df)
        labels = [col for col in df.columns if 'M o d e l' not in col]
        print(labels)

    for col in labels:
        df[col] = pd.to_numeric(df[col], errors='coerce') * 100 + offset

    max_value = max(df[labels].max())

    # Calculate the number of groups
    n_groups = len(labels)

    # Set the width for each bar and calculate positions
    bar_width = args.bar_width
    index = np.arange(len(df) * n_groups + (n_groups - 1) * int(bar_width * 2.5))  # Adjust spacing based on bar width

    # Colors and hatches
    #colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    #colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', 'blue', 'orange', 'green', '#e7298a', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    colors = ['#1f78b4', '#a6cee3', '#33a02c', '#b2df8a', '#e31a1c', '#fb9a99', '#ff7f00', '#fdbf6f', '#6a3d9a', '#cab2d6', '#b15928', '#ffff99', '#003f5c', '#444e86', '#955196', '#dd5182']
    hatches = ['', '/', '', '-', '', '+', '', '\\', '', '|', '', '.', '', 'x', '', 'o', '', 'O', '', '*']

    # Create the bar plots
    fig, ax = plt.subplots(figsize=(20, 5))  # Increase horizontal size
    for i in range(len(df)):
        for j, label in enumerate(labels):
            ax.bar(index[i + j * (len(df) + int(bar_width * 2.5))], df[label].iloc[i], bar_width,
                   color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])

    # Add labels, title, and legend
    #ax.set_xlabel('Models per Attack', fontsize=15, fontweight='bold')
    ax.set_ylabel('Adversarial Success (%)', fontsize=15, fontweight='bold')
    ax.set_title('Adversarial Success Rates', fontsize=20, fontweight='bold')
    ax.set_xticks([np.mean(index[i * (len(df) + int(bar_width * 2.5)):(i + 1) * (len(df) + int(bar_width * 2.5))]) for i in range(n_groups)])
    ax.set_ylim(0, max_value * 1.1)  # Y-Achse dynamisch auf maximalen Wert einstellen
    ax.set_xticklabels(labels, fontsize=15, fontweight='bold')

    # Create legend with unique labels and patterns
    handles = [matplotlib.patches.Patch(facecolor=colors[i % len(colors)], hatch=hatches[i % len(hatches)]) for i in range(len(df))]
    print(df)
    legend_labels = df.iloc[:,0]#[f'Index {i+1}' for i in range(len(df))]
    ax.legend(handles, legend_labels, loc='best', ncol=4, fontsize='x-large')

    ax.axhline(y=offset, color='r', linestyle='--', linewidth=1)

    yticks = ax.get_yticks()
    print(yticks)
    new_ytick_labels = [""] + [float(tick - offset) for tick in yticks[1:]]
    ax.set_yticklabels(new_ytick_labels)

    ax.grid(True, axis='y')

    # Adjust layout and display the plot
    fig.tight_layout()
    fig.savefig('barplot.png')
    plt.close(fig)
    #plt.show()
    #plt.close()

if __name__ == '__main__':
    # command to start: python automaticBars.py --path Adv_Auswertung_Extracted.xlsx  --filter " S"  --header 2 --bar_width 1.0 --sheet LeNet_strong
    main()
#%%
