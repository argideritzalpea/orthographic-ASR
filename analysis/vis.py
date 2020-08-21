import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

df = pd.read_csv('compare_with_gold_context_PhonFeatureCategories - Sheet9.csv')
df['diff'] = df.apply(lambda row: row['Percent 3c correct'] - row['Percent 3b correct'], axis=1)
df = df.sort_values('diff')

manner = df[df['Type category'] == "manner"]
manner = manner[manner['Phonemic feature'] != "distributed"]

place = df[df['Type category'] == "place"]
major = df[df['Type category'] == "major"]
laryngeal = df[df['Type category'] == "laryngeal"]




sns.set(rc={'figure.figsize':(8,6)})

# Manner
sns.set(style="whitegrid")
custom_palette = sns.diverging_palette(10, 250)
for idx, q in enumerate(set(place["diff"])):
    if idx == 2:
        custom_palette[idx] = 'lavender'
g = sns.barplot(x="Phonemic feature", y="diff", data=manner, palette=custom_palette)
#for i, bar in enumerate(g.patches):
#    bar.set_color("C{}".format(i%1))
g.set_ylabel("% improvement of 3c over 3b")
g.set_xticklabels(g.get_xticklabels(), rotation=50, ha="right")
g.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.set_title('Manner', fontsize=16)
figure = g.get_figure()
figure.tight_layout()
figure.savefig('manner.png')

# Place
sns.set(style="whitegrid")
custom_palette = sns.color_palette("Blues")
for idx, q in enumerate(set(place["diff"])):
    if q < 0:
        custom_palette[idx] = 'r'
g = sns.barplot(x="Phonemic feature", y="diff", data=place, palette=custom_palette)
g.set_ylabel("% improvement of 3c over 3b")
g.set_xticklabels(g.get_xticklabels(), rotation=50, ha="right")
g.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.set_title('Place', fontsize=16)
figure = g.get_figure()
figure.tight_layout()
figure.savefig('place.png')
# Major
sns.set(style="whitegrid")
g = sns.barplot(x="Phonemic feature", y="diff", data=major, palette="Blues")
g.set_ylabel("% improvement of 3c over 3b")
g.set_xticklabels(g.get_xticklabels(), rotation=50, ha="right")
g.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.set_title('Sonority', fontsize=16)
figure = g.get_figure()
figure.tight_layout()
figure.savefig('sonority.png')
# Laryngeal
sns.set(style="whitegrid")
g = sns.barplot(x="Phonemic feature", y="diff", data=laryngeal, palette="Blues")
g.set_ylabel("% improvement of 3c over 3b")
g.set_xticklabels(g.get_xticklabels(), rotation=50, ha="right")
g.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.set_title('Laryngeal', fontsize=16)
figure = g.get_figure()
figure.tight_layout()
figure.savefig('laryngeal.png')


sns.barplot(place['Phonemic feature'], place['diff'], palette=np.array(pal[::-1])[rank])


sns.barplot(df['Phonemic feature'], df['diff'], hue=df["Type category"], )


# library
import matplotlib.pyplot as plt

# Make fake dataset
height = df['diff']
bars = df['Phonemic feature']
# Choose the width of each bar and their positions
df["Normalized for area"] = np.int_(df['Disagreements'] * (1-(np.abs(df['diff'])/np.sum(df['diff']))))
width = df["Normalized for area"]/100000

# Compute pie slices
plt.bar(df['Phonemic feature'], df['diff'], width=width*20, alpha=0.5)








# Colorbar - difference
"""
norm = plt.Normalize(manner['Disagreements_perc'].min(), manner['Disagreements_perc'].max())
sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
sm.set_array([])
pal = sns.color_palette("Greens_d", len(perc))
rank = perc.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
plt.figure(figsize=(6,6))
ax = sns.barplot(manner['Phonemic feature'], manner['diff'], hue=manner['Disagreements'], dodge=False, palette=sns.color_palette('Reds', len(manner)))
ax.figure.colorbar(sm)
ax.get_legend().remove()
ax.set_ylabel('Percent deviation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
ax.axhline(y=0, color='k')
plt.text(1.3, 0.5, '% of disagreements of aligned phonemes in class', horizontalalignment='center',
    verticalalignment='center', transform=ax.transAxes, rotation=270)
plt.title("Manner", fontsize=16)
plt.show()
"""
