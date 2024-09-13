import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches

# Beispiel-Daten
labels = ['Gruppe 1', 'Gruppe 2', 'Gruppe 3']
data1 = [20, 34, 30, 22, 28, 35, 40, 45, 50, 55, 60]
data2 = [25, 32, 34, 27, 30, 38, 42, 48, 52, 58, 62]
data3 = [30, 30, 35, 32, 36, 40, 44, 50, 54, 60, 65]

# Anzahl der Gruppen
n_groups = len(labels)

# Positionen der Gruppen auf der x-Achse mit Abstand zwischen den Gruppen
index = np.arange(11 * n_groups + (n_groups - 1) * 2)  # 2 spaces between groups

# Breite der Balken
bar_width = 0.8

# Farben und Schraffierungen
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
hatches = ['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# Erstellen der Balken
fig, ax = plt.subplots()
for i in range(11):
    ax.bar(index[i], data1[i], bar_width, label=f'Gruppe 1 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])
    ax.bar(index[13 + i], data2[i], bar_width, label=f'Gruppe 2 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])
    ax.bar(index[26 + i], data3[i], bar_width, label=f'Gruppe 3 - {i+1}', color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])

# Hinzufügen von Labels, Titel und Legende
ax.set_xlabel('Gruppen')
ax.set_ylabel('Werte')
ax.set_title('Bardiagramm mit 3 Gruppen von Datenpunkten')
ax.set_xticks([5, 18, 31])  # Mittelwert der Balkenpositionen jeder Gruppe
ax.set_xticklabels(labels)
# Erstellen der Legende mit 11 verschiedenen Labels und Mustern
handles = []
legend_labels = []
for i in range(11):
    handles.append(matplotlib.patches.Patch(facecolor = colors[i % len(colors)], hatch=hatches[i % len(hatches)]))
    legend_labels.append(f'Index {i+1}')

ax.legend(handles, legend_labels)

# Layout anpassen und Diagramm anzeigen
fig.tight_layout()
plt.show()
plt.close()

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Beispiel-Daten
# labels = ['Gruppe 1', 'Gruppe 2', 'Gruppe 3']
# data1 = [20, 34, 30, 22, 28, 35, 40, 45, 50, 55, 60]
# data2 = [25, 32, 34, 27, 30, 38, 42, 48, 52, 58, 62]
# data3 = [30, 30, 35, 32, 36, 40, 44, 50, 54, 60, 65]
#
# # Anzahl der Gruppen
# n_groups = len(labels)
#
# # Positionen der Gruppen auf der x-Achse mit Abstand zwischen den Gruppen
# index = np.arange(11 * n_groups + (n_groups - 1) * 2)  # 2 spaces between groups
#
# # Breite der Balken
# bar_width = 0.8
#
# # Erstellen der Balken
# fig, ax = plt.subplots()
# bar1 = ax.bar(index[:11], data1, bar_width, label='Gruppe 1')
# bar2 = ax.bar(index[13:24], data2, bar_width, label='Gruppe 2')  # 13:24 to account for 2 spaces
# bar3 = ax.bar(index[26:], data3, bar_width, label='Gruppe 3')  # 26: to account for 2 spaces
#
# # Hinzufügen von Labels, Titel und Legende
# ax.set_xlabel('Gruppen')
# ax.set_ylabel('Werte')
# ax.set_title('Bardiagramm mit 3 Gruppen von Datenpunkten')
# ax.set_xticks([5, 18, 31])  # Mittelwert der Balkenpositionen jeder Gruppe
# ax.set_xticklabels(labels)
# ax.legend()
#
# # Layout anpassen und Diagramm anzeigen
# fig.tight_layout()
# plt.show()
# plt.close()
#%%
