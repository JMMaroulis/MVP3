import Methods_Poisson
import numpy
import random
import sys
import matplotlib.pyplot as plt
import copy
import numpy
import scipy
import gc
from operator import itemgetter
import sys
from scipy.optimize import minimize
import operator


width = int(sys.argv[1])
height = int(sys.argv[2])
length = int(sys.argv[3])
# 0 = euler, 1 = gauss-seidel
algorithm = int(sys.argv[4])
omega = float(sys.argv[5])
# 0 = do not, 1 = write to file
write_to_file = int(sys.argv[6])
# 0 = electric, 1 = magnetic
field_type = int(sys.argv[7])
tolerance = float(sys.argv[8])


# initialise charge grid
charge_grid = numpy.zeros((width + 2, height + 2, length + 2))
charge_storage_grid = copy.deepcopy(charge_grid)


# SET CHARGES:
# wire
for i in range(0, length+2):
    Methods_Poisson.set_cell(charge_grid, int(width/2), int(height/2), i, 100)



"""
# random point charges
choice = range(0, length)

for i in range(0, 100):
    x = random.choice(choice)
    y = random.choice(choice)
    z = random.choice(choice)
    Methods_Poisson.set_cell(charge_grid, x, y, z, random.uniform(-20, 20))
"""


# runs algorithm on potential grid until convergence
# produces electric field grid form result
# (omega input value requires singe-element numpy array input; required to get minimisation algorithm working)
# (it's quite dumb, but there's no sensible workaround)
def converge_electric_potential(omega_value):
    global potential_grid, potential_change, field_grid, potential_storage_grid, field_storage_grid

    # initialise potential_grids
    potential_grid = numpy.zeros((width+2, height+2, length+2))
    potential_storage_grid = copy.deepcopy(charge_grid)

    # initialise electric field grid
    field_grid = numpy.zeros((width+2, height+2, length+2), dtype=tuple)
    # convert field_grid from scalar to (x,y,z) vector tuples
    for i in range(0, width+2):
        for j in range(0, height+2):
            for k in range(0, length+2):
                    Methods_Poisson.set_cell(field_grid, i, j, k, (0, 0, 0))

    field_storage_grid = copy.deepcopy(field_grid)

    # set values, (re-)initialise potential / field grid
    omega = omega_value[0]

    # pass values to method class
    Methods_Poisson.set_things(width, height, length, charge_grid, charge_storage_grid,
                               potential_grid, potential_storage_grid, field_grid, field_storage_grid, omega)

    # number of steps taken until convergence
    n = 0

    # loop until potentials converged
    gc.disable()

    # arbitrary starting value to get the while loop going
    # error bound set to 10**-4
    potential_change = 10
    while potential_change >= tolerance:
        if algorithm == 0:
            potential_grid, potential_change = copy.deepcopy(Methods_Poisson.jacobi_update_potential_sweep())
        if algorithm == 1:
            potential_grid, potential_change = copy.deepcopy(Methods_Poisson.gauss_seidel_update_potential_sweep())
        print(potential_change)
        n += 1
    gc.collect()

    # generate electric field from converged potential
    field_grid = Methods_Poisson.generate_electric_field()

    print('omega = ', omega)
    print('n = ', n)
    print('----------')
    return n


def converge_magnetic_potential(omega_value):
    global potential_grid, potential_change, field_grid, potential_storage_grid, field_storage_grid

    # initialise potential_grids
    potential_grid = numpy.zeros((width+2, height+2, length+2))
    potential_storage_grid = copy.deepcopy(charge_grid)

    # initialise magnetic field grid
    field_grid = numpy.zeros((width+2, height+2, length+2), dtype=tuple)
    # convert field_grid from scalar to (x,y,z) vector tuples
    for i in range(0, width+2):
        for j in range(0, height+2):
            for k in range(0, length+2):
                    Methods_Poisson.set_cell(field_grid, i, j, k, (0, 0, 0))

    field_storage_grid = copy.deepcopy(field_grid)

    # set values, (re-)initialise potential / field grid
    omega = omega_value[0]

    # pass values to method class
    Methods_Poisson.set_things(width, height, length, charge_grid, charge_storage_grid,
                               potential_grid, potential_storage_grid, field_grid, field_storage_grid, omega)

    # number of steps taken until convergence
    n = 0

    # loop until potentials converged
    gc.disable()

    # arbitrary starting value to get the while loop going
    # error bound set to 10**-4
    potential_change = 10
    while potential_change >= tolerance:
        if algorithm == 0:
            potential_grid, potential_change = copy.deepcopy(Methods_Poisson.jacobi_update_potential_sweep())
        if algorithm == 1:
            potential_grid, potential_change = copy.deepcopy(Methods_Poisson.gauss_seidel_update_potential_sweep())
        print(potential_change)
        n += 1
    gc.collect()

    # generate electric field from converged potential
    field_grid = Methods_Poisson.generate_magnetic_field()

    print('omega = ', omega)
    print('n = ', n)
    print('----------')
    return n

#################### OMEGA OPTIMISATION SECTION

# ALGORITHMIC APPROACH

"""

omega_value = numpy.array([1.3])
print(omega_value)
values = scipy.optimize.minimize(converge_electric_potential, omega_value, args=(), method='Powell', jac=None, hess=None, hessp=None,
                                 bounds=((0.0001, None), (0.0001, None), (10 ** 3, 10 ** 3)), constraints=(), tol=None,
                                 callback=None, options={'disp': True})

"""
"""

# BRUTE-FORCE APPROACH
omega_txt = open("omega_test.txt", "w")
omega = 1.0
n_dict = dict()
while omega <= 1.4:
    omega_value = numpy.array([omega])
    n_sweeps = converge_electric_potential(omega_value)
    n_dict[omega] = n_sweeps
    omega += 0.05

    # write to file
    omega_txt.write('%s' % omega)
    omega_txt.write(' ')
    omega_txt.write('%s \n' % n_sweeps)
omega_txt.close()


print(n_dict)
omega_n_sorted = sorted(n_dict.items(), key=operator.itemgetter(1))
print(omega_n_sorted)


"""

########
omega_value = numpy.array([omega])
if field_type == 0:
    print('converging electric potential...')
    print('tolerance:', tolerance)
    converge_electric_potential(omega_value)
elif field_type == 1:
    print('converging magnetic potential...')
    print('tolerance:', tolerance)
    converge_magnetic_potential(omega_value)
#########



########################## FILE WRITING SECTION

if write_to_file == 1:
    # ELECTRIC FIELD OUTPUTTING
    # convert electric field tuples into lists for plotting
    x_pos_list = list()
    y_pos_list = list()
    z_pos_list = list()
    x_val_list = list()
    y_val_list = list()
    z_val_list = list()
    magnitude_list = list()
    wire_distance_list = list()

    gc.collect()
    # unpack field tuples into lists
    for i in range(1, width+1):
        for j in range(1, height+1):
            for k in range(1, length+1):
                field_tuple = Methods_Poisson.get_cell(field_grid, i, j, k)
                # get rid of tiny field vectors
                if (field_tuple[0]**2 + field_tuple[1]**2 + field_tuple[2]**2) ** 0.5 >= 10**-3.0:
                    x_val_list.append(field_tuple[0])
                    y_val_list.append(field_tuple[1])
                    z_val_list.append(field_tuple[2])
                    x_pos_list.append(i)
                    y_pos_list.append(j)
                    z_pos_list.append(k)
                    wire_distance = ((i-int(width/2.0))**2.0) + ((j-int(height/2.0))**2.0)
                    wire_distance **= 0.5
                    wire_distance_list.append(wire_distance)
    gc.collect()

    # normalise vector lengths
    for i in range(0, len(x_val_list)):
        x = x_val_list[i]
        y = y_val_list[i]
        z = z_val_list[i]

        magnitude = (x**2.0 + y**2.0 + z**2.0) ** 0.5
        x /= magnitude
        y /= magnitude
        z /= magnitude

        x_val_list[i] = x
        y_val_list[i] = y
        z_val_list[i] = z
        magnitude_list.append(magnitude)

    # shove data into .txt file for gnuplot
    if field_type == 0:
        electric_field = open("electric_field.txt", "w")
    if field_type == 1:
        electric_field = open("magnetic_field.txt", "w")

    for i in range(0, len(x_pos_list)):

        x = x_pos_list[i]
        y = y_pos_list[i]
        z = z_pos_list[i]
        a = x_val_list[i]
        b = y_val_list[i]
        c = z_val_list[i]
        dist = wire_distance_list[i]
        magnitude = magnitude_list[i]
        # print(x,y,x,a,b,c)

        electric_field.write('%s' % x)
        electric_field.write(' ')
        electric_field.write('%s' % y)
        electric_field.write(' ')
        electric_field.write('%s' % z)
        electric_field.write(' ')
        electric_field.write('%s' % a)
        electric_field.write(' ')
        electric_field.write('%s' % b)
        electric_field.write(' ')
        electric_field.write('%s' % c)
        electric_field.write(' ')
        electric_field.write('%s' % dist)
        electric_field.write(' ')
        electric_field.write('%s \n' % magnitude)

    electric_field.close()

    # POTENTIAL FIELD PLOTTING

    del x_pos_list
    del y_pos_list
    del z_pos_list
    del x_val_list
    del y_val_list
    del z_val_list
    del wire_distance_list

    x_pos_list = list()
    y_pos_list = list()
    z_pos_list = list()
    pot_val_list = list()
    wire_distance_list = list()

    gc.collect()

    # unpack grid into various lists
    for i in range(1, width+1):
        for j in range(1, height+1):
            for k in range(1, length+1):
                pot = Methods_Poisson.get_cell(potential_grid, i, j, k)

                # for purposes of plotting, ignore uninteresting values
                if abs(pot) >= 10**-4:
                    pot_val_list.append(pot)
                    x_pos_list.append(i)
                    y_pos_list.append(j)
                    z_pos_list.append(k)
                    wire_distance = ((i-int(width/2.0))**2.0) + ((j-int(height/2.0))**2.0)
                    wire_distance **= 0.5
                    wire_distance_list.append(wire_distance)

    # shove data into .txt file for gnuplot
    if field_type == 0:
        potential_field = open("electric_potential_field.txt", "w")
    if field_type == 1:
        potential_field = open("magnetic_potential_field.txt", "w")

    for i in range(0, len(x_pos_list)):

        x = x_pos_list[i]
        y = y_pos_list[i]
        z = z_pos_list[i]
        p = pot_val_list[i]
        dist = wire_distance_list[i]

        potential_field.write('%s' % x)
        potential_field.write(' ')
        potential_field.write('%s' % y)
        potential_field.write(' ')
        potential_field.write('%s' % z)
        potential_field.write(' ')
        potential_field.write('%s' % p)
        potential_field.write(' ')
        potential_field.write('%s \n' % dist)

    potential_field.close()

    # splot "potential_field.txt" using 1:2:3:(log($4)) with circles palette
    # splot "electric_field.txt" with vectors
