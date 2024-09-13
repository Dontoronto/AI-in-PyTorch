import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Laden der Excel-Datei
file_path = 'beispiel.xlsx'
df = pd.read_excel(file_path, sheet_name='ResNet_like_training', header=[0, 1])

# Anzeigen der ursprünglichen Spaltennamen
print("Ursprüngliche Spaltennamen:")
print(df)



# Umbenennen der Spalten
# df.columns = [
#     'Model', 'DeepFool S', 'DeepFool O', 'DeepFool F-P',
#     'OnePixel S', 'OnePixel O', 'OnePixel F-P', 'PGD S', 'PGD O', 'PGD F-P'
# ]
df.columns = [' '.join(col).strip() for col in df.columns.values]

# Entfernen der ersten Zeile, die die Header wiederholt
#df = df.drop(0)

offset = 1

print(df)

# Konvertieren Sie die Daten in numerische Werte, falls sie als Strings importiert wurden
df['DeepFool S'] = pd.to_numeric(df['DeepFool S'], errors='coerce') * 100 + offset
df['OnePixel S'] = pd.to_numeric(df['OnePixel S'], errors='coerce') * 100 + offset
df['PGD S'] = pd.to_numeric(df['PGD S'], errors='coerce') * 100 + offset




max_value = max(df['DeepFool S'].max(), df['OnePixel S'].max(), df['PGD S'].max())



filter_string = 'S'

labels = [col for col in df.columns if filter_string in col]
max_value = max(df[labels].max())
print(labels)

# max_value = max(df['DeepFool S'].max(), df['OnePixel S'].max(), df['PGD S'].max())

n_groups = len(labels)

# Set the width for each bar
# Positionen der Gruppen auf der x-Achse mit Abstand zwischen den Gruppen
index = np.arange(11 * n_groups + (n_groups - 1) * 5)  # 2 spaces between groups

# Breite der Balken
bar_width = 1

# Farben und Schraffierungen
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
hatches = ['', '/', '\\', '|', '-', '+', '', 'o', 'O', '.', '']



# Erstellen der Balken
fig, ax = plt.subplots()
for i in range(11):
    ax.bar(index[i], df['DeepFool S'][i], bar_width, label=f'Gruppe 1 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])
    ax.bar(index[16 + i], df['PGD S'][i], bar_width, label=f'Gruppe 2 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])
    ax.bar(index[32 + i], df['OnePixel S'][i], bar_width, label=f'Gruppe 3 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])

# Hinzufügen von Labels, Titel und Legende
ax.set_xlabel('Gruppen')
ax.set_ylabel('Werte')
ax.set_title('Bardiagramm mit 3 Gruppen von Datenpunkten')
ax.set_xticks([5, 22, 38])  # Mittelwert der Balkenpositionen jeder Gruppe
ax.set_ylim(0, max_value * 1.1)  # Y-Achse dynamisch auf maximalen Wert einstellen
ax.set_xticklabels(labels)
# Erstellen der Legende mit 11 verschiedenen Labels und Mustern
handles = []
legend_labels = []
for i in range(11):
    handles.append(matplotlib.patches.Patch(facecolor = colors[i % len(colors)], hatch=hatches[i % len(hatches)]))
    legend_labels.append(f'Index {i+1}')

ax.legend(handles, legend_labels, loc='best', ncols=4, fontsize='small')
ax.axhline(y=offset, color='r', linestyle='--', linewidth=2)

yticks = ax.get_yticks()
new_yticks = yticks - offset
new_ytick_labels = [""] + [int(tick - offset) for tick in yticks[1:]]
ax.set_yticklabels(new_ytick_labels)

# Layout anpassen und Diagramm anzeigen
fig.tight_layout()
plt.show()
plt.close()


#%%
