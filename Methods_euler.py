import random
import numpy
import copy

dt = 0.1
dx = 0.5
a = 0.1
m = 0.1
kappa = 0.1
width = int()
height = int()


# transfer things into Methods class for weird reasons
def set_things(w, h):
    global width, height
    width = w
    height = h


# element getter, can deal with looping edges
def get_cell(state, x_pos, y_pos):
    return state[x_pos % width, y_pos % height]


# element setter, can deal with looping edges
def set_cell(state, x_pos, y_pos, new_value):
    state[x_pos % width, y_pos % height] = new_value


# Euler Algorithm Step
def euler_update_sweep(state, storage_state):

    for i in range(0, width):
        for j in range(0, height):

            # calculate new phi value, put in storage
            phi_new = get_cell(state, i, j) + get_dphi_dt(state, i, j)
            set_cell(storage_state, i, j, phi_new)

    # retrieve from storage
    state = copy.deepcopy(storage_state)
    return state


##################################


# BOTTOM UP
def get_del2_phi(state, x_pos, y_pos):

    del2_phi = 0
    del2_phi += get_cell(state, x_pos+1, y_pos)
    del2_phi += get_cell(state, x_pos-1, y_pos)
    del2_phi += get_cell(state, x_pos, y_pos+1)
    del2_phi += get_cell(state, x_pos, y_pos-1)
    del2_phi -= 4.0 * get_cell(state, x_pos, y_pos)
    del2_phi /= dx ** 2.0

    return del2_phi


# calculate mu element from phi
def get_mu(state, x_pos, y_pos):

    mu = 0
    mu -= a * get_cell(state, x_pos, y_pos)
    mu += a * (get_cell(state, x_pos, y_pos)**3)
    mu -= kappa * get_del2_phi(state, x_pos, y_pos)

    return mu


# calculate del^2.mu
def get_del2_mu(state, x_pos, y_pos):

    del2_mu = 0
    del2_mu += get_mu(state, x_pos+1, y_pos)
    del2_mu += get_mu(state, x_pos-1, y_pos)
    del2_mu += get_mu(state, x_pos, y_pos+1)
    del2_mu += get_mu(state, x_pos, y_pos-1)
    del2_mu -= 4.0 * get_mu(state, x_pos, y_pos)
    del2_mu *= dt * (dx ** (-2.0))

    return del2_mu


# get change in phi element
def get_dphi_dt(state, x_pos, y_pos):

    dphi_dt = m * get_del2_mu(state, x_pos, y_pos)

    return dphi_dt


# update phi element
def get_phi_new(state, x_pos, y_pos):

    phi_old = get_cell(state, x_pos, y_pos)
    phi_new = phi_old + get_dphi_dt(state, x_pos, y_pos)

    return phi_new


# for oil/water conservation checking
def sum_oil_water(state):

    total = 0

    for i in range(0, width):
        for j in range(0, height):
            total += get_cell(state, i, j)

    return total


# calculate (del.phi)^2
def get_del_phi2(state, x_pos, y_pos):

    del_phi2 = 0
    del_phi2 += (get_cell(state, x_pos+1, y_pos) - get_cell(state, x_pos-1, y_pos)) ** 2.0
    del_phi2 += (get_cell(state, x_pos, y_pos+1) - get_cell(state, x_pos, y_pos-1)) ** 2.0
    del_phi2 *= dx ** (-2.0)
    del_phi2 /= 4.0

    return del_phi2


def free_energy(state):

    free_energy = 0.0

    for i in range(0, width):
        for j in range(0, height):

            free_energy -= (a/2.0) * (get_cell(state, i, j)**2.0)
            free_energy += (a/4.0) * (get_cell(state, i, j)**4.0)
            free_energy += (kappa/2.0) * get_del_phi2(state, i, j)

    return free_energy



