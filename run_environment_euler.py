import Methods_euler
import numpy
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gc
import copy


# user argument inputs
width = int(sys.argv[1])
height = int(sys.argv[2])
oil_proportion = float(sys.argv[3])
water_proportion = float(sys.argv[4])
visualisation = int(sys.argv[5])
num_sweeps = int(sys.argv[6])


# Initialisation

# pass values to methods class
Methods_euler.set_things(width, height)

# INITIALISE PHI
# grid of phi values, initialising at zero
phi_grid = numpy.zeros((width, height))
phi_storage_grid = numpy.zeros((width, height))

# set all phi to +-0.5 with noise
for i in range(0, width):
    for j in range(0, height):

        list = [0.0]
        x = random.choice(list) + numpy.random.uniform(-0.1, 0.1)
        Methods_euler.set_cell(phi_grid, i, j, x)

# Methods.set_cell(phi_grid, 10, 9, .001)


# check oil/water conservation
print(Methods_euler.sum_oil_water(phi_grid))

if visualisation == 0:
    gc.disable()
    # run loop
    t = 0
    free_energy_list = []
    for i in range(0, num_sweeps):
        phi_grid = Methods_euler.euler_update_sweep(phi_grid, phi_storage_grid)
        t += 1
        if t == 1:
            free_energy_list.append(Methods_euler.free_energy(phi_grid))
            t = 0
    gc.collect()

    # write to file
    free_energy_file = open("free_energy.txt", "w")
    n = 0
    for x in free_energy_list:
        free_energy_file.write('%s' % x)
        free_energy_file.write(' ')
        free_energy_file.write('%s \n' % n)
        n += 1
    free_energy_file.close()

    # check oil/water conservation
    print(Methods_euler.sum_oil_water(phi_grid))

    image = plt.imshow(phi_grid, interpolation="none")
    plt.colorbar()
    plt.show()


# ANIMATION ATTEMPT
if visualisation == 1:

    gc.disable()

    # keeps animation happy
    def update(state):
        mat.set_data(state)
        return mat

    # generator to keep animation happy
    def data_gen_euler():
        while True:
            yield generate_data_euler()

    def generate_data_euler():
        global phi_grid
        phi_grid = Methods_euler.euler_update_sweep(phi_grid, phi_storage_grid)
        return phi_grid

    # code animates happily until visualisation stopped manually
    fig, ax = plt.subplots()
    mat = ax.matshow(generate_data_euler())
    plt.colorbar(mat)
    ani = animation.FuncAnimation(fig, update, data_gen_euler, interval=1)

    plt.show()
