import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 伪造的氨基酸序列
sequence = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWME"

fig, ax = plt.subplots(figsize=(12, 2))

# 长条背景表示整条序列
ax.add_patch(Rectangle((0, 0.25), len(sequence), 0.5, color='lightgrey'))

# 每10个氨基酸标一个字母
for i, aa in enumerate(sequence):
    if i % 5 == 0:
        ax.text(i + 0.2, 0.5, aa, fontsize=9, va='center')

# 可标结构域、功能位点
ax.add_patch(Rectangle((5, 0.25), 10, 0.5, color='skyblue', alpha=0.7))
ax.text(5, 0.9, 'Domain A', fontsize=9)

ax.set_xlim(0, len(sequence))
ax.set_ylim(0, 1.5)
ax.axis('off')
plt.show()