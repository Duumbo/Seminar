import matplotlib.pyplot as plt
import numpy as np

mvmc_16 = np.genfromtxt("16sites.mvmc")
imp_16 = np.genfromtxt("timerun/params4")

fig, ax = plt.subplots()

ax.plot(range(0, len(mvmc_16[:,0])), mvmc_16[:,0])
ax.plot(range(0, len(imp_16[:,0])), imp_16[:,0])
ax.set_xlabel("Itération d'optimisation")
ax.set_ylabel(r"$\langle E\rangle$")
ax.set_title(r"$N_{MC} = 10k, N_{OPT} = 1000, N_{S}=16$")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.60, 0.95, rf"$E_{{MVMC}}={np.round(np.mean(mvmc_16[-15:-1,0]), 4)}$", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.text(0.60, 0.75, rf"$E_{{CONV}}={np.round(np.mean(imp_16[-15:-1,0]), 4)}$", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

fig.savefig("16sites.png")

mvmc_36 = np.genfromtxt("36sites.mvmc")
imp_36 = np.genfromtxt("timerun/params6")

fig, ax = plt.subplots()

ax.plot(range(0, len(mvmc_36[:,0])), mvmc_36[:,0])
ax.plot(range(0, len(imp_36[:,0])), imp_36[:,0])
ax.set_xlabel("Itération d'optimisation")
ax.set_ylabel(r"$\langle E\rangle$")
ax.set_title(r"$N_{MC} = 10k, N_{OPT} = 1000, N_{S}=36$")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.60, 0.95, rf"$E_{{MVMC}}={np.round(np.mean(mvmc_36[-15:-1,0]), 4)}$", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.text(0.60, 0.75, rf"$E_{{CONV}}={np.round(np.mean(imp_36[-15:-1,0]), 4)}$", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

fig.savefig("36sites.png")

fig, ax = plt.subplots(layout='constrained')

imp_time = np.genfromtxt("time_file")
mvmc_time = np.genfromtxt("time6.mvmc")

x = np.arange(len(imp_time))
width = 0.25  # the width of the bars
multiplier = 0


offset = width * multiplier
rects = ax.bar(x + offset, imp_time, width, label="Temps de convergence")
ax.bar_label(rects, padding=3)
multiplier += 1
offset = width * multiplier
rects = ax.bar(x + offset, mvmc_time, width, label="Temps (mVMC)")
ax.bar_label(rects, padding=3)

ax.set_xticks(x + width, [r"$N=4$", r"$N=16$", "$N=36$"])
ax.legend(loc='upper left', ncols=2)
fig.savefig("Barplot.png")

plt.show()
