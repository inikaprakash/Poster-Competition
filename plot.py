import numpy as np
import matplotlib.pyplot as plt

models = [
    "CatBoost",
    "Random Forest",
    "Black–Scholes",
    "LSTM",
    "XGBoost"
]

rmse_iv = [15.3074, 15.2861, 19.0363, 17.3995, 43.7340]
rmse_hv = [14.4594, 14.6260, 18.1357, 17.8574, 47.3520]

background = "#F1EAEF"
dark_purple = "#240E3C"
lavender = "#AB7ED9"

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor(background)
ax.set_facecolor(background)

y = np.arange(len(models))
bar_height = 0.34

bars_iv = ax.barh(
    y - bar_height / 2,
    rmse_iv,
    bar_height,
    label="Implied volatility (IV)",
    color=dark_purple
)

bars_hv = ax.barh(
    y + bar_height / 2,
    rmse_hv,
    bar_height,
    label="Historical volatility (HV)",
    color=lavender
)

for bars in [bars_iv, bars_hv]:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=9,
            color=dark_purple
        )

ax.set_yticks(y)
ax.set_yticklabels(
    models,
    fontsize=11,
    color=dark_purple
)

ax.set_xlabel(
    "Root Mean Squared Error (RMSE)",
    fontsize=11,
    color=dark_purple,
    labelpad=10
)

ax.invert_yaxis()
ax.set_xlim(0, 53)

ax.xaxis.grid(
    True,
    linestyle="--",
    alpha=0.25,
    color=dark_purple
)

ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(
    axis="both",
    length=0,
    colors=dark_purple
)

ax.set_title(
    "Option Pricing Accuracy: Implied vs Historical Volatility",
    fontsize=14,
    fontweight="bold",
    color=dark_purple,
    pad=38
)

legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=2,
    frameon=False,
    fontsize=10
)

for text in legend.get_texts():
    text.set_color(dark_purple)

plt.tight_layout()
plt.subplots_adjust(top=0.83)

plt.savefig(
    "volatility_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=background
)

plt.show()
