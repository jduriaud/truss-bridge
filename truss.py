"""
truss.py
========

The direct stiffness method for a two-dimensional pin-jointed truss.

A truss is a set of straight bars connected at frictionless joints. Each bar
carries only an axial force (tension or compression). Bending is neglected.
Every node has two degrees of freedom, its horizontal and vertical
displacement, so a truss with N nodes has 2N degrees of freedom, and the static
problem reduces to the linear system

    K u = F ,

where K is the global stiffness matrix, u the vector of nodal displacements,
and F the vector of nodal forces.

A truss is described here by two objects:

    nodes : an (n_nodes, 2) array of node coordinates,
    bars  : a list of (i, j) pairs giving the node indices at each bar's ends.

The module assembles K, solves for the displacements under given loads and
supports, recovers the axial forces, and computes the natural modes of
vibration. Node and degree-of-freedom indices are zero-based: node k owns
degrees of freedom 2k (horizontal) and 2k + 1 (vertical).
"""

import numpy as np
from scipy.linalg import eigh


# --------------------------------------------------------------------------- #
# Stiffness
# --------------------------------------------------------------------------- #
def element_stiffness(node_i, node_j, E, A):
    """Return the 4x4 stiffness matrix of one bar, in global coordinates.

    The bar runs from node_i to node_j, each a pair of coordinates, and has
    Young's modulus E and cross-sectional area A. With direction cosines
    c = cos(theta) and s = sin(theta), the axial stiffness E A / L is projected
    onto the global axes to give the matrix acting on (u_i, v_i, u_j, v_j).
    """
    (x_i, y_i) = node_i
    (x_j, y_j) = node_j
    length = np.hypot(x_j - x_i, y_j - y_i)
    c = (x_j - x_i) / length
    s = (y_j - y_i) / length

    block = np.array([
        [c * c,  c * s, -c * c, -c * s],
        [c * s,  s * s, -c * s, -s * s],
        [-c * c, -c * s,  c * c,  c * s],
        [-c * s, -s * s,  c * s,  s * s],
    ])
    return (E * A / length) * block


def assemble_stiffness(nodes, bars, E, A):
    """Assemble the global stiffness matrix by summing the bar contributions.

    Each bar adds its 4x4 matrix to the four blocks of K indexed by the degrees
    of freedom of the two nodes it connects.
    """
    n_dof = 2 * len(nodes)
    K = np.zeros((n_dof, n_dof))
    for i, j in bars:
        element = element_stiffness(nodes[i], nodes[j], E, A)
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(dofs, dofs)] += element
    return K


# --------------------------------------------------------------------------- #
# Solving the static problem
# --------------------------------------------------------------------------- #
def free_dofs(n_dof, fixed_dofs):
    """Return the indices of the degrees of freedom that are free to move."""
    return np.setdiff1d(np.arange(n_dof), fixed_dofs)


def solve_displacements(K, forces, fixed_dofs):
    """Solve K u = F with the supported degrees of freedom held at zero.

    The rows and columns of the fixed degrees of freedom are removed, the
    reduced system is solved, and the result is placed back into a full vector
    with zeros at the supports.
    """
    free = free_dofs(len(forces), fixed_dofs)
    displacements = np.zeros(len(forces))
    displacements[free] = np.linalg.solve(K[np.ix_(free, free)], forces[free])
    return displacements


def support_reactions(K, displacements, forces):
    """Return the reaction forces, non-zero only at the supported nodes."""
    return K @ displacements - forces


def is_stable(K, fixed_dofs):
    """Return True if the supported truss is rigid, i.e. its reduced stiffness
    matrix is non-singular. A singular matrix signals a mechanism: the
    structure can deform at no cost and the static problem has no unique
    solution.
    """
    free = free_dofs(K.shape[0], fixed_dofs)
    reduced = K[np.ix_(free, free)]
    return np.linalg.matrix_rank(reduced) == reduced.shape[0]


# --------------------------------------------------------------------------- #
# Internal forces and loads
# --------------------------------------------------------------------------- #
def axial_force(node_i, node_j, displacement_i, displacement_j, E, A):
    """Return the axial force in one bar, positive in tension.

    The bar's elongation is the projection of the relative nodal displacement
    onto its axis. Multiplying by the axial stiffness E A / L gives the force.
    """
    (x_i, y_i) = node_i
    (x_j, y_j) = node_j
    length = np.hypot(x_j - x_i, y_j - y_i)
    c = (x_j - x_i) / length
    s = (y_j - y_i) / length

    elongation = (
        c * (displacement_j[0] - displacement_i[0])
        + s * (displacement_j[1] - displacement_i[1])
    )
    return (E * A / length) * elongation


def all_axial_forces(nodes, bars, displacements, E, A):
    """Return the axial force in every bar, in the order of `bars`."""
    forces = []
    for i, j in bars:
        displacement_i = displacements[2 * i:2 * i + 2]
        displacement_j = displacements[2 * j:2 * j + 2]
        forces.append(axial_force(nodes[i], nodes[j], displacement_i, displacement_j, E, A))
    return np.array(forces)


def self_weight_forces(nodes, bars, linear_density, g=9.81):
    """Return the nodal force vector from the bars' own weight.

    Each bar's weight is split equally between its two end nodes and applied
    downward.
    """
    n_dof = 2 * len(nodes)
    forces = np.zeros(n_dof)
    for i, j in bars:
        length = np.hypot(*(nodes[j] - nodes[i]))
        weight = linear_density * length * g
        forces[2 * i + 1] -= weight / 2
        forces[2 * j + 1] -= weight / 2
    return forces


# --------------------------------------------------------------------------- #
# Vibration
# --------------------------------------------------------------------------- #
def lumped_mass_matrix(nodes, bars, linear_density):
    """Return the diagonal (lumped) mass matrix of the truss.

    Half of each bar's mass is assigned to each of its end nodes, and applied
    equally to that node's horizontal and vertical degrees of freedom.
    """
    n_dof = 2 * len(nodes)
    diagonal = np.zeros(n_dof)
    for i, j in bars:
        length = np.hypot(*(nodes[j] - nodes[i]))
        half_mass = linear_density * length / 2
        for node in (i, j):
            diagonal[2 * node] += half_mass
            diagonal[2 * node + 1] += half_mass
    return np.diag(diagonal)


def natural_modes(K, M, fixed_dofs, n_modes):
    """Return the lowest natural frequencies and their mode shapes.

    The generalised eigenvalue problem K phi = omega^2 M phi is solved on the
    free degrees of freedom. The eigenvalues give the natural frequencies
    f = omega / (2 pi), returned in ascending order. Each mode shape is placed
    back into a full displacement vector with zeros at the supports.
    """
    free = free_dofs(K.shape[0], fixed_dofs)
    eigenvalues, eigenvectors = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

    shapes = np.zeros((K.shape[0], n_modes))
    for mode in range(n_modes):
        shapes[free, mode] = eigenvectors[:, mode]
    return frequencies[:n_modes], shapes
