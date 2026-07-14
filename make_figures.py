"""Regenerate every figure and the animation in figures/.

    python make_figures.py

The script also serves as a worked example of how to drive truss.py.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

import truss

# ---- a calm, consistent house style --------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})
UNDEFORMED = "#c3c7cc"
NODE = "#22303f"
LOAD = "#e76f51"
AXIAL_CMAP = "coolwarm"
FIG = "figures"

# ---- the bridge: geometry, supports, and material ------------------------- #
HEIGHT = 5 * np.sqrt(3)                      # equilateral triangles of side 10 m
NODES = np.array([
    [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0],
    [5.0, HEIGHT], [15.0, HEIGHT], [25.0, HEIGHT],
])
BARS = [(0, 1), (1, 2), (2, 3), (3, 6), (5, 6), (4, 5),
        (0, 4), (1, 4), (1, 5), (2, 5), (2, 6)]
BARS_NO_TRUSS = [(0, 1), (1, 2), (2, 3), (3, 6), (5, 6), (4, 5), (0, 4)]
DECK_NODES = [0, 1, 2, 3]                    # the four nodes carrying traffic
FIXED_DOFS = [0, 1, 6, 7]                    # nodes N1 and N4 pinned to the banks

E = 2.0e11                                   # Young's modulus, N/m^2
A = 0.01                                     # cross-section, m^2
LINEAR_DENSITY = 7850 * A                    # steel bar, kg/m
VEHICLE_WEIGHT = 1000 * 9.81                 # N

K = truss.assemble_stiffness(NODES, BARS, E, A)


# ---- drawing helpers ------------------------------------------------------ #
def deformed(displacement, amplify):
    return NODES + amplify * displacement.reshape(-1, 2)


def auto_amplify(displacement, target=4.0):
    largest = np.abs(displacement).max()
    return target / largest if largest > 0 else 1.0


def draw_bars(ax, points, bars, axial=None, norm=None, lw=3):
    if axial is None:
        for i, j in bars:
            ax.plot(*zip(points[i], points[j]), color=UNDEFORMED, lw=lw, zorder=1)
        return None
    colormap = plt.get_cmap(AXIAL_CMAP)
    for value, (i, j) in zip(axial, bars):
        ax.plot(*zip(points[i], points[j]), color=colormap(norm(value)), lw=lw, zorder=2)
    return ScalarMappable(norm=norm, cmap=colormap)


def style_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-3, 33)
    ax.set_ylim(-8, 12)


# ---- 1. Geometry ---------------------------------------------------------- #
def figure_geometry():
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    draw_bars(ax, NODES, BARS)
    ax.scatter(NODES[:, 0], NODES[:, 1], color=NODE, s=45, zorder=3)
    for k, (xp, yp) in enumerate(NODES):
        ax.annotate(f"N{k + 1}", (xp, yp), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=NODE)
    for support in (0, 3):
        ax.scatter(*NODES[support], marker="s", s=140, color=LOAD, zorder=4)
    for loaded in (1, 2):
        ax.annotate("", xy=(NODES[loaded, 0], NODES[loaded, 1] - 3),
                    xytext=(NODES[loaded, 0], NODES[loaded, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=LOAD, lw=2))
    style_axes(ax)
    ax.set_title("The truss: seven nodes, eleven bars, pinned at N1 and N4")
    fig.savefig(f"{FIG}/geometry.png")
    plt.close(fig)
    print("  geometry.png")


# ---- 2. Static case, bars coloured by axial force ------------------------- #
def figure_static():
    forces = np.zeros(2 * len(NODES))
    forces[2 * 1 + 1] = -1000
    forces[2 * 2 + 1] = -1000
    displacement = truss.solve_displacements(K, forces, FIXED_DOFS)
    axial = truss.all_axial_forces(NODES, BARS, displacement, E, A) / 1000
    amplify = auto_amplify(displacement)
    points = deformed(displacement, amplify)
    limit = np.abs(axial).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    draw_bars(ax, NODES, BARS, lw=1.5)
    mappable = draw_bars(ax, points, BARS, axial=axial, norm=norm)
    ax.scatter(points[:, 0], points[:, 1], color=NODE, s=25, zorder=3)
    style_axes(ax)
    ax.set_title(f"Static load: deformation (×{amplify:.0e}) and axial forces")
    bar = fig.colorbar(mappable, ax=ax, shrink=0.8)
    bar.set_label("axial force (kN)   — compression   + tension")
    fig.savefig(f"{FIG}/static_forces.png")
    plt.close(fig)
    print("  static_forces.png")


# ---- 3. Self-weight ------------------------------------------------------- #
def figure_self_weight():
    forces = truss.self_weight_forces(NODES, BARS, LINEAR_DENSITY)
    displacement = truss.solve_displacements(K, forces, FIXED_DOFS)
    axial = truss.all_axial_forces(NODES, BARS, displacement, E, A) / 1000
    amplify = auto_amplify(displacement)
    points = deformed(displacement, amplify)
    limit = np.abs(axial).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    draw_bars(ax, NODES, BARS, lw=1.5)
    mappable = draw_bars(ax, points, BARS, axial=axial, norm=norm)
    ax.scatter(points[:, 0], points[:, 1], color=NODE, s=25, zorder=3)
    style_axes(ax)
    ax.set_title(f"Under self-weight: deformation (×{amplify:.0e}) and axial forces")
    bar = fig.colorbar(mappable, ax=ax, shrink=0.8)
    bar.set_label("axial force (kN)   — compression   + tension")
    fig.savefig(f"{FIG}/self_weight.png")
    plt.close(fig)
    print("  self_weight.png")


# ---- 4. Natural modes ----------------------------------------------------- #
def figure_modes():
    M = truss.lumped_mass_matrix(NODES, BARS, LINEAR_DENSITY)
    frequencies, shapes = truss.natural_modes(K, M, FIXED_DOFS, 6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), constrained_layout=True)
    for mode, ax in enumerate(axes.flat):
        displacement = shapes[:, mode]
        points = deformed(displacement, auto_amplify(displacement, target=3.5))
        draw_bars(ax, NODES, BARS, lw=1.2)
        draw_bars(ax, points, BARS, lw=2.2)
        for i, j in BARS:
            ax.plot(*zip(points[i], points[j]), color=LOAD, lw=2.2, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 33)
        ax.set_ylim(-6, 15)
        ax.set_title(f"Mode {mode + 1}: f = {frequencies[mode]:.1f} Hz", fontsize=10)
        ax.grid(alpha=0.15)
    fig.suptitle("First six natural modes of vibration", fontsize=13)
    fig.savefig(f"{FIG}/modes.png")
    plt.close(fig)
    print("  modes.png")


# ---- 5. The mechanism when the truss is removed --------------------------- #
def figure_mechanism():
    K_no = truss.assemble_stiffness(NODES, BARS_NO_TRUSS, E, A)
    free = truss.free_dofs(K_no.shape[0], FIXED_DOFS)
    reduced = K_no[np.ix_(free, free)]
    _, _, right_vectors = np.linalg.svd(reduced)
    mechanism = np.zeros(K_no.shape[0])
    mechanism[free] = right_vectors[-1]            # smallest singular value: zero-energy motion
    points = deformed(mechanism, auto_amplify(mechanism, target=4.0))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    draw_bars(ax, NODES, BARS_NO_TRUSS, lw=1.5)
    for i, j in BARS_NO_TRUSS:
        ax.plot(*zip(points[i], points[j]), color=LOAD, lw=2.5, ls="--", zorder=2)
    ax.scatter(NODES[:, 0], NODES[:, 1], color=NODE, s=25, zorder=3)
    style_axes(ax)
    ax.set_title("Without the diagonal bars: a zero-energy mechanism")
    fig.savefig(f"{FIG}/mechanism.png")
    plt.close(fig)
    print("  mechanism.png")


# ---- 6. Hero animation: a vehicle crossing the bridge --------------------- #
def moving_load(x_car, weight):
    forces = np.zeros(2 * len(NODES))
    deck_x = NODES[DECK_NODES, 0]
    for segment in range(len(DECK_NODES) - 1):
        x_left = deck_x[segment]
        x_right = deck_x[segment + 1]
        if x_left <= x_car <= x_right:
            t = (x_car - x_left) / (x_right - x_left)
            forces[2 * DECK_NODES[segment] + 1] -= weight * (1 - t)
            forces[2 * DECK_NODES[segment + 1] + 1] -= weight * t
            break
    return forces


def animation_vehicle():
    positions = np.linspace(0, 30, 90)
    displacements = []
    axials = []
    for x_car in positions:
        forces = moving_load(x_car, VEHICLE_WEIGHT)
        u = truss.solve_displacements(K, forces, FIXED_DOFS)
        displacements.append(u)
        axials.append(truss.all_axial_forces(NODES, BARS, u, E, A) / 1000)

    amplify = 30000.0
    limit = np.abs(axials).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    colormap = plt.get_cmap(AXIAL_CMAP)

    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    style_axes(ax)
    ax.set_title("A vehicle crossing the bridge")
    draw_bars(ax, NODES, BARS, lw=1.2)
    lines = [ax.plot([], [], lw=3, zorder=2)[0] for _ in BARS]
    car = ax.scatter([], [], marker="s", s=90, color=NODE, zorder=5)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax, shrink=0.8)
    bar.set_label("axial force (kN)   — compression   + tension")

    def update(frame):
        points = deformed(displacements[frame], amplify)
        for line, value, (i, j) in zip(lines, axials[frame], BARS):
            line.set_data(*zip(points[i], points[j]))
            line.set_color(colormap(norm(value)))
        x_car = positions[frame]
        segment = min(int(x_car // 10), 2)
        t = (x_car - 10 * segment) / 10
        node_a = DECK_NODES[segment]
        node_b = DECK_NODES[segment + 1]
        car.set_offsets([(1 - t) * points[node_a] + t * points[node_b]])
        return lines + [car]

    animation = FuncAnimation(fig, update, frames=len(positions), blit=True)
    animation.save(f"{FIG}/vehicle.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"  vehicle.gif  ({len(positions)} frames)")


if __name__ == "__main__":
    figure_geometry()
    figure_static()
    figure_self_weight()
    figure_modes()
    figure_mechanism()
    animation_vehicle()
    print("done.")
