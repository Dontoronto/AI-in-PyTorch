import matplotlib.pyplot as plt
import numpy as np

# Beispiel-Daten: Ersetzen Sie diese Daten durch Ihre eigenen
categories = ['192x3x3x3', '64x192x1x1', '96x192x1x1', '128x96x3x3', '32x192x1x1',
              '16x192x1x1', '208x96x3x3', '96x480x1x1', '192x480x1x1', '48x16x3x3',
              '48x16x3x3', '64x480x1x1', '160x832x1x1', '320x160x3x3', '32x832x1x1',
              '128x32x3x3', '128x128x3x3', '128x832x1x1', '32x832x1x1', '64x480x1x1',
              '256x832x1x1']
values = [
    [1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5], [2, 3, 4, 5, 6], [1, 2, 2.5, 3, 4],
    [1.2, 2.1, 3.1, 4.1, 5.1], [1.3, 2.2, 3.2, 4.2, 5.2], [1.4, 2.3, 3.3, 4.3, 5.3],
    [1.5, 2.4, 3.4, 4.4, 5.4], [1.6, 2.5, 3.5, 4.5, 5.5], [1.7, 2.6, 3.6, 4.6, 5.6],
    [1.8, 2.7, 3.7, 4.7, 5.7], [1.9, 2.8, 3.8, 4.8, 5.8], [2, 2.9, 3.9, 4.9, 5.9],
    [2.1, 3, 4, 5, 6], [2.2, 3.1, 4.1, 5.1, 6.1], [2.3, 3.2, 4.2, 5.2, 6.2],
    [2.4, 3.3, 4.3, 5.3, 6.3], [2.5, 3.4, 4.4, 5.4, 6.4], [2.6, 3.5, 4.5, 5.5, 6.5],
    [2.7, 3.6, 4.6, 5.6, 6.6], [2.8, 3.7, 4.7, 5.7, 6.7]
]

x = np.arange(len(categories))  # Position der Kategorien auf der x-Achse
n_values = len(values[0])  # Anzahl der Messwerte pro Messpunkt
width = 0.1  # Breite der Balken

fig, ax = plt.subplots(figsize=(20, 6))

# Erstellen der Balken für jede Messwertkategorie
for i in range(n_values):
    offset = (i - n_values / 2) * width + width / 2
    ax.bar(x + offset, [v[i] for v in values], width, label=f'Kategorie {i+1}')

# Hinzufügen von Text und Labels
ax.set_xlabel('Kategorien')
ax.set_ylabel('Werte (%)')
ax.set_title('Vergleich von kategorischen Werten')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_ylim(0, 10)  # Y-Achse auf Bereich 0-100% setzen
ax.legend()

# Prozentzahlen über den Balken anzeigen (optional)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte (vertikal) über den Balken
                    textcoords="offset points",
                    ha='center', va='bottom')

fig.tight_layout()

plt.show()
