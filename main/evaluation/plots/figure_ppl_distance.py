import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# NABirds
x = [20, 50, 100, 500]
y_diffae = [383.04, 364.02, 350.05, 335.72]
y_diffusegae = [370.73, 355.24, 345.67, 330.72]

x_ticks = [20, 50, 100, 500]
y_ticks = [320, 340, 360, 380, 390]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

axes.plot(x, y_diffae, linestyle='-', color='#8A949B', marker='.', label='Diff-AE', linewidth=0.8, )
axes.plot(x, y_diffusegae, linestyle='-', color='#fd5f00', marker='.', label='DiffuseGAE', linewidth=0.8, )
# axes[0].plot(x, y_gzsnet, linestyle='-', color='#fd5f00', marker='.', label='Diff-AE', linewidth=0.8,)

# plt.axis("on")
font_prop = fm.FontProperties(fname='Palatino.ttf')
# 设置x、y轴标签
axes.set_ylabel("Perceptual Path Length", fontname='Palatino')
axes.set_xlabel("Number of timestep", fontname='Palatino')

# 设置y轴的刻度
axes.set_yticks(y_ticks)

axes.set_xticks(x)

axes.legend(prop={'family': 'Palatino'})
axes.grid(color='lightgray', linestyle='--', linewidth=0.8)

# Set the font for tick labels
axes.set_xticklabels(x_ticks, fontproperties=font_prop)
axes.set_yticklabels(y_ticks, fontproperties=font_prop)

plt.tight_layout()
# Save the plot as a PDF file
plt.savefig('PPL distance.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)

