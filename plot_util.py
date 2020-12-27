import os
import matplotlib.pyplot as plt
import numpy as np

latestRe = 100
Imax = 41
Jmax = 41

dx = 5.0 / (Imax-1)
dy = 1.0 / (Jmax-1)

dirName = os.path.dirname(__file__)
fileName = os.path.join(
    dirName, f'results/correct/Re-{latestRe}_dRe-50_I-{Imax}_J-{Jmax}.dat')


nu = Imax * (Jmax - 1)
nv = Jmax * (Imax - 1)

results = np.loadtxt(fileName, skiprows=2)
u = np.array(results[0:nu])
v = np.array(results[nu:nu+nv])
p = np.array(results[nu+nv:])

u = np.resize(u, (Jmax-1, Imax))
v = np.resize(v, (Imax, Jmax-1))
p = np.resize(p, (Imax-1, Jmax-1))


# Calculation of x and y coordinates
# ===========================================================================#
x = np.zeros(Imax-1)

y = np.zeros(Jmax-1)

for i in range(Imax-1):
    x[i] = i*dx+dx/2

for i in range(Jmax-1):
    y[i] = i*dy+dy/2
# ===========================================================================#


# Interpolation of u and v primitive variables to cell center
# ============================================================================#
uP_c = np.zeros((Jmax-1, Imax-1))

vP_c = np.zeros((Jmax-1, Imax-1))

for i in range(Jmax-1):
    for j in range(Imax-1):
        uP_c[i, j] = (u[i, j]+u[i, j+1])/2
        vP_c[i, j] = (v[i, j]+v[i+1, j])/2


# fig, _axs = plt.subplots(nrows=4, ncols=1)

# axs = _axs.flatten()
# # Streamplot
# strm = axs[0].streamplot(x, y, uP_c, vP_c, color="k",
#                          linewidth=0.5, density=(4, 2))
# axs[0].set_title('Streamlines at Re='+str(latestRe))
# axs[0].set_aspect(1)

# # u velocity plot
# uplot = axs[1].contourf(x, y, uP_c)
# axs[1].set_title('u velocity distribution at Re='+str(latestRe))
# fig.colorbar(uplot, ax=axs[1])
# axs[1].set_aspect(1)

# # v velocity plot
# vplot = axs[2].contourf(x, y, vP_c)
# axs[2].set_title('v velocity distribution at Re='+str(latestRe))
# fig.colorbar(vplot, ax=axs[2])
# axs[2].set_aspect(1)

# # Pressure plot
# pplot = axs[3].contourf(x, y, p)
# axs[3].set_title('p distribution at Re='+str(latestRe))
# axs[3].set_aspect(1)
# fig.colorbar(pplot, ax=axs[3])

# # Figure options
# fig.subplots_adjust(hspace=0.35)
# plt.show()


def plot_x3(grid, Re, dRe=[50], line_style=['solid', 'dashed', 'dotted', 'dashdot', 'dashdashdot']):
    dirName = os.path.dirname(__file__)

    if len(dRe) == 1:
        # plot analysed values
        for i, IJmax in enumerate(grid):
            Imax = IJmax
            Jmax = IJmax
            fileName = os.path.join(
                dirName, f'results/correct/Re-{Re}_dRe-{dRe[0]}_I-{Imax}_J-{Imax}.dat')

            nu = Imax * (Jmax - 1)
            nv = Jmax * (Imax - 1)

            results = np.loadtxt(fileName, skiprows=2)
            u = np.array(results[0:nu])

            u = np.resize(u, (Jmax-1, Imax))
            u_3 = u[:, int((Imax-1)/5.0*3.0)]
            y_3 = np.linspace(0, 1, (Jmax-1))

            plt.plot(u_3, y_3, color="k",
                     ls=line_style[i], linewidth=0.5, label=f'{Imax}x{Jmax}')
    else:
        # plot different delta Reynolds values:
        for i, dReynolds in enumerate(dRe):
            Imax = grid[0]
            Jmax = grid[0]
            fileName = os.path.join(
                dirName, f'results/correct/Re-{Re}_dRe-{dReynolds}_I-{Imax}_J-{Jmax}.dat')

            nu = Imax * (Jmax - 1)
            nv = Jmax * (Imax - 1)

            results = np.loadtxt(fileName, skiprows=2)
            u = np.array(results[0:nu])

            u = np.resize(u, (Jmax-1, Imax))
            u_3 = u[:, int((Imax-1)/5.0*3.0)]
            y_3 = np.linspace(0, 1, (Jmax-1))

            plt.plot(u_3, y_3, color="k",
                     ls=line_style[i], linewidth=0.5, label=f'$\Delta Re={dReynolds}$')

    # plot comparison values:
    if Re == 0:
        expResultName = os.path.join(
            dirName, r'results/X=3_PROFILE_STOKES.dat')
    if Re == 100:
        expResultName = os.path.join(
            dirName, r'results/X=3_PROFILE_RE=100.dat')

    results = np.loadtxt(expResultName, skiprows=10)
    rows = np.where(results[:, 0] == 3)
    filteredResults = results[rows]

    sortedResults = filteredResults[filteredResults[:, 1].argsort()]
    plt.plot(sortedResults[:, 2], sortedResults[:, 1],
             linewidth=0.5, label='Ninova')

    plt.xlabel('u velocity')
    plt.ylabel('y')
    plt.title(f'u Velocity Comparison. Re={Re}, x@3')
    plt.axes().set_aspect(1)
    plt.legend()
    plt.show()


plot_x3([201], 100, dRe=[10, 25, 50, 100])
