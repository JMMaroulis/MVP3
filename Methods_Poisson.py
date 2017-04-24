import random
import numpy
import copy

dt = 1.0
dx = 1.0


# transfer things into Methods class for weird reasons
def set_things(w, h, l, c, c_s, p, p_s, f, f_s, o):
    global width, height, length, charge_grid, charge_storage_grid, potential_grid, potential_storage_grid, field_grid, field_storage_grid, omega
    width = w
    height = h
    length = l
    charge_grid = c
    charge_storage_grid = c_s
    potential_grid = p
    potential_storage_grid = p_s
    field_grid = f
    field_storage_grid = f_s
    omega = o


# element getter, NOT PERIODIC
def get_cell(state, x_pos, y_pos, z_pos):
    return state[x_pos, y_pos, z_pos]


# element setter, NOT PERIODIC
def set_cell(state, x_pos, y_pos, z_pos, new_value):
    state[x_pos, y_pos, z_pos] = new_value
    return state


# Jacobi Algorithm
def jacobi_update_potential_sweep():
    global potential_grid, potential_storage_grid

    # ranges set so that halo of zero-valued boundary remains intact
    for i in range(1, width+1):
        for j in range(1, height+1):
            for k in range(1, length+1):

                potential = get_potential(potential_grid, i, j, k)
                set_cell(potential_storage_grid, i, j, k, potential)

    # get potential cell-by-cell change summation
    potential_change = get_potential_change()

    potential_grid = copy.deepcopy(potential_storage_grid)
    return potential_grid, potential_change


def gauss_seidel_update_potential_sweep():
    global potential_grid, potential_storage_grid

    potential_storage_grid = copy.deepcopy(potential_grid)

    # ranges set so that halo of zero-valued boundary remains intact
    for i in range(1, width+1):
        for j in range(1, height+1):
            for k in range(1, length+1):

                potential = get_potential_omega(potential_grid, i, j, k)
                set_cell(potential_grid, i, j, k, potential)

    # get potential cell-by-cell change summation
    potential_change = get_potential_change()

    return potential_grid, potential_change


def generate_electric_field():
    global field_grid, field_storage_grid

    for i in range(1, width+1):
        for j in range(1, height+1):
            for k in range(1, length+1):
                field_element = get_electric_field(i, j, k)
                set_cell(field_grid, i, j, k, field_element)

    return field_grid

def generate_magnetic_field():
    global field_grid, field_storage_grid

    for i in range(1, width + 1):
        for j in range(1, height + 1):
            for k in range(1, length + 1):

                field_element = get_magnetic_field(i, j, k)
                set_cell(field_grid, i, j, k, field_element)

    return field_grid

########################


# define electric potential element from charge grid
def get_potential(potential_grid, x_pos, y_pos, z_pos):

    potential = 0
    potential += get_cell(potential_grid, x_pos+1, y_pos, z_pos)
    potential += get_cell(potential_grid, x_pos-1, y_pos, z_pos)
    potential += get_cell(potential_grid, x_pos, y_pos+1, z_pos)
    potential += get_cell(potential_grid, x_pos, y_pos-1, z_pos)
    potential += get_cell(potential_grid, x_pos, y_pos, z_pos+1)
    potential += get_cell(potential_grid, x_pos, y_pos, z_pos-1)
    potential += get_cell(charge_grid, x_pos, y_pos, z_pos)
    potential *= dt * (dx ** -2.0)

    potential /= 6.0

    return potential


# define electric potential element from charge grid using omega
def get_potential_omega(potential_grid, x_pos, y_pos, z_pos):

    current_potential = get_cell(potential_grid, x_pos, y_pos, z_pos)
    new_potential = get_potential(potential_grid, x_pos, y_pos, z_pos)
    potential_change = new_potential - current_potential

    potential_change *= omega
    potential = current_potential + potential_change

    return potential


# define electric field element from potential field grid
def get_electric_field(x_pos, y_pos, z_pos):

    field_x, field_y, field_z = 0, 0, 0

    field_x += get_cell(potential_grid, x_pos+1, y_pos, z_pos)
    field_x -= get_cell(potential_grid, x_pos-1, y_pos, z_pos)
    field_x /= -dx

    field_y += get_cell(potential_grid, x_pos, y_pos+1, z_pos)
    field_y -= get_cell(potential_grid, x_pos, y_pos-1, z_pos)
    field_y /= -dx

    field_z += get_cell(potential_grid, x_pos, y_pos, z_pos+1)
    field_z -= get_cell(potential_grid, x_pos, y_pos, z_pos-1)
    field_z /= -dx

    field_tuple = (field_x, field_y, field_z)

    return field_tuple


def get_magnetic_field(x_pos, y_pos, z_pos):

    field_x, field_y, field_z = 0, 0, 0
    daz_dy = 0.0
    daz_dx = 0.0

    # del x A = (dAz/dy)ex - (dAz/dx)ey

    # (dAz/dy)
    # print(get_cell(potential_grid, x_pos, y_pos+1, z_pos))
    daz_dy += get_cell(potential_grid, x_pos, y_pos+1, z_pos)
    daz_dy -= get_cell(potential_grid, x_pos, y_pos-1, z_pos)
    daz_dy /= 2.0 * dx
    field_x = daz_dy

    # (dAz/dx)
    daz_dx += get_cell(potential_grid, x_pos+1, y_pos, z_pos)
    daz_dx -= get_cell(potential_grid, x_pos-1, y_pos, z_pos)
    daz_dx /= 2.0 * dx
    field_y = -daz_dx

    field_tuple = (field_x, field_y, field_z)

    return field_tuple


def get_potential_change():
    potential_change = 0.0
    for i in range(1, width + 1):
        for j in range(1, height + 1):
            for k in range(1, length + 1):
                potential_old = get_potential(potential_grid, i, j, k)
                potential_new = get_potential(potential_storage_grid, i, j, k)
                potential_change += abs(potential_old - potential_new)

    return potential_change






