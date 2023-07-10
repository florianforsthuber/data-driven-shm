"""Helper module to create new damage locations and sizes.
Writes a .csv file with damage location and sizes to the cwd.
Execute this script before running Abaqus script 'create_submodels.py'.

Author: student k1256205@students.jku.at
Created: 15/06/2022
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv


def damage_locations(rng, N_new_damages, size_xy, damage_radius_range, sensor_locations, sensor_size):
    """
    Randomly generate damage locations and sizes with respect to the positional constraints of the submodel.
    Hardcoded assumptions resulting in constraints:
    - major strains and stresses are small at one diameter distance from the hole. Healthy distance of damage to edges.
    - At the non-driven edge the hole can be closer. Distance: 1 radius.
    - Sensors: HBM RY93-6/120 -> 6x6 mm size
    :param rng: random Generator -> np best practice
    :param N_new_damages: number of damages to generate
    :param size_xy: dimensions of submodel in submodel coords.
    :param damage_radius_range: radius interval of damage in mm
    :param sensor_locations: locations of sensors in submodel coords
    :param sensor_size: edge length of one sensor
    :return: list of damage locations (submodel coords.) and sizes
    """
    damages = []
    for i in range(N_new_damages):
        # generate one damage location and size
        while True:
            # generate random radius in range
            radius = rng.uniform(damage_radius_range[0], damage_radius_range[1])
            radius = np.around(radius, decimals=2)
            # generate random position inside submodel area
            pos_x = rng.uniform(0, size_xy[0])
            pos_x = np.around(pos_x, decimals=2)
            pos_y = rng.uniform(0, size_xy[1])
            pos_y = np.around(pos_y, decimals=2)
            # check boundary conditions edges
            if pos_x < 3 * radius or pos_y < 3 * radius:
                # lower and left constraint violated
                continue
            if pos_x > size_xy[0] - 3 * radius:
                # right constraint violated
                continue
            if pos_y > size_xy[1] - 2 * radius:
                # upper constraint violated
                continue
            # check boundary conditions sensors
            boolean_list = [np.sqrt((sensor_xy[0] - pos_x)**2 + (sensor_xy[1] - pos_y)**2) < sensor_size * 1/np.sqrt(2) + radius for sensor_xy in sensor_locations]
            if any(boolean_list):
                # sensor position constraint violated
                continue

            # random position passed constraint check
            damages.append({'name': 'DS' + str(i), 'x': pos_x, 'y': pos_y, 'radius': radius})
            break
    return damages

def visualize_damage_location(size_xy, sensor_locations, sensor_size, damage_location):
    """
    Visualize the random damage and the positional constraints.
    :param size_xy: dimensions of submodel in submodel coords.
    :param sensor_locations: locations of sensors in submodel coords
    :param damage_location: dict {x,y,r}
    :return: None
    """
    fig = plt.figure()
    ax = fig.gca()
    pos_x = damage_location['x']
    pos_y = damage_location['y']
    radius = damage_location['radius']

    # visualize sensor constraints
    for sensor_xy in sensor_locations:
        ax.add_patch(patches.Rectangle((sensor_xy[0] - sensor_size / 2, sensor_xy[1] - sensor_size / 2),
                                       sensor_size,
                                       sensor_size,
                                       facecolor='red',
                                       alpha=0.5,
                                       ec='black'))
    # visualize edge constraints
    # left edge
    ax.add_patch(patches.Rectangle((0, 0),
                                   2 * radius, size_xy[1],
                                   facecolor='red', alpha=0.5, ec='black'))
    # lower edge
    ax.add_patch(patches.Rectangle((0, 0),
                                   size_xy[0], 2 * radius,
                                   facecolor='red', alpha=0.5, ec='black'))
    # right edge
    ax.add_patch(patches.Rectangle((size_xy[0], 0),
                                   -(2 * radius), size_xy[0],
                                   facecolor='red', alpha=0.5, ec='black'))
    # upper edge
    ax.add_patch(patches.Rectangle((0, size_xy[1]),
                                   size_xy[0], -radius,
                                   facecolor='red', alpha=0.5, ec='black'))

    # visualize damage location
    ax.add_patch(patches.Circle((pos_x, pos_y), radius, facecolor='blue', alpha=0.5, ec='black'))

    ax.set(xlim=(0, size_xy[0]), ylim=(0, size_xy[1]))
    ax.set_aspect('equal')
    plt.show()

def dict_to_csv(data, filepath):
    """
    Procedure to convert list of dicts to .csv file
    :param data: list of dictionaries
    :param filepath: filepath/filename.csv to create/overwrite
    :return: None
    """
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def csv_to_dict(filepath):
    """
    Procedure to convert .csv file to list of dictionaries.
    Each row -> dicitonary represents one observation
    :param filepath: filepath/filename.csv to load
    :return: list of dictionaries
    """
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # fields = reader.next()
        for row in reader:
            data.append(row)
    return data


if __name__ == '__main__':
    # random generator
    # seed should guarantee that the damage locations do not change
    # if this module is executed again -> reproducible relationship between
    # damage state naming and damage location
    rng = default_rng(seed=40)
    #rng = default_rng()

    # submodel dimensions
    size_xy = (220, 195.5)
    damage_radius_range = [5, 12]
    sensor_locations_global = np.array([[195, 352.5], [250, 352.5], [305, 352.5],
                                  [195, 297.5], [250, 297.5], [305, 297.5],
                                  [195, 242.5], [250, 242.5], [305, 242.5]])
    sensor_size = 6
    N_new_damages = 100

    # transform sensor locations to sensor coordinate system
    delta_x = np.full(9, 140)
    delta_y = np.full(9, 187.5)
    sensor_locations = sensor_locations_global
    sensor_locations[:, 0] -= delta_x
    sensor_locations[:, 1] -= delta_y

    # generate data
    damage_locs = damage_locations(rng, N_new_damages, size_xy, damage_radius_range, sensor_locations, sensor_size)

    # save results to csv file
    dict_to_csv(damage_locs, filepath='../scripts/Damage_locations.csv')

    damage_para = csv_to_dict(filepath='../scripts/Damage_locations.csv')

    for row in damage_para[0:4]:
        print(f"name: {row['name']}, x: {row['x']}, y: {row['y']}, radius: {row['radius']}")

    # visualize
    for each in damage_locs[0:4]:
        visualize_damage_location(size_xy, sensor_locations, sensor_size, each)
