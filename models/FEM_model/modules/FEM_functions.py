"""Module with functions for handling FEM model
Module holds general functions for dealing with everything connected to the abaqus FEM model.

Import this as a module: from FEM_functions import <specific function>
    * requirement: FEM_functions.py needs to be in the same directory as module that imports it.
    -> functions will be available in namespace.

simulate_submodels.py is the origin of most functions. For now it is left as is.
Author: student k1256205@students.jku.at
Created: 01/08/2022
"""
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import json
import csv
from tqdm import tqdm


def extract_nodal_results(new_odb_filepath, instance, step):
    """
    Extract the nodal information from one .odb file of submodel simulation result.

    1. Creates a new directory in data/raw/simulated according to damage state,
    if it does not exist yet
    2. writes the necessary information for node extraction in form of a dict into a JSON File,
    located in the same directory as the abaqus script
    3. invokes abaqus to run "extract_node_data.py"
    -> .csv file is written to the target directory specified in json file
    :param new_odb_filepath: newly created directory of (one specific) submodel loadcase simulation result
    :param instance: name of the instance of the submodel to extract nodal data from
    (SUB-CORE-1, SUB-SKIN-TOP-1, SUB-SKIN-BOTTOM-1)
    :param step: name of the simulation step which holds the results
    :return: target_dir: directory where extracted nodal results .csv is saved to
    """

    current_dir = Path.cwd()
    # create a new directory in data/raw/simulated/ -> damage state -> nodal_data
    parent_dir = current_dir.parents[2]
    assert parent_dir.stem == "work"
    new_DS_dir = new_odb_filepath.parent.stem
    target_dir = parent_dir / 'data' / 'raw' / 'simulated' / new_DS_dir / 'nodal_data'
    try:
        target_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"No new directory created. Directory {target_dir} already exists.")

    # change current working directory to abaqus scripts
    os.chdir(current_dir.parent / 'scripts')

    # define data to be dumped to json file
    extraction_settings = {'odb_filepath': str(new_odb_filepath),
                           'target_dir': str(target_dir),
                           'Step': step,
                           'Instance': instance,
                           'sP_layer': 'Top_ply'
                           }

    # create a json file extract_node_data_support.json
    with open("extract_node_data_support.json", "w") as write_file:
        json.dump(extraction_settings, write_file)

    try:
        print(f"Starting extraction of node data...")
        os.system('abaqus cae noGUI=extract_node_data.py')
        print(f"Extraction of node data COMPLETED")
    finally:
        os.chdir(current_dir)

    return target_dir


def virtual_strain_data(extracted_nodal_data, strain_scaling="microstrain"):
    """
    Interpolation of virtual strain data at given sensor positions in bottom instance of the submodel.
    Assumption: 0/45/90 strain gauge
    :param extracted_nodal_data: Directory which stores .csv files with extracted nodal results for one submodel type
    :param strain_scaling: str -> flag controlling the scaling
    :return: None
    """
    # interpolate virtual strain at given sensor positions.
    # check if directory is of type Path
    if not isinstance(extracted_nodal_data, Path):
        extracted_nodal_data = Path(extracted_nodal_data)
    #nodal_data_dir = extracted_nodal_data / 'nodal_data'

    # set microstrain scale
    if strain_scaling == "microstrain":
        scaling = 10**6
    else:
        scaling = 1

    sensor_information = {1: {"x": 195, "y": 352.5, "desc_y": "V_1_1", "desc_x": "H_1_1", "desc_xi": "VH_1_1"},
                          2: {"x": 250, "y": 352.5, "desc_y": "V_2_1", "desc_x": "H_1_2", "desc_xi": "VH_1_2"},
                          3: {"x": 305, "y": 352.5, "desc_y": "V_3_1", "desc_x": "H_1_3", "desc_xi": "VH_1_3"},
                          4: {"x": 195, "y": 297.5, "desc_y": "V_1_2", "desc_x": "H_2_1", "desc_xi": "VH_2_1"},
                          5: {"x": 250, "y": 297.5, "desc_y": "V_2_2", "desc_x": "H_2_2", "desc_xi": "VH_2_2"},
                          6: {"x": 305, "y": 297.5, "desc_y": "V_3_2", "desc_x": "H_2_3", "desc_xi": "VH_2_3"},
                          7: {"x": 195, "y": 242.5, "desc_y": "V_1_3", "desc_x": "H_3_1", "desc_xi": "VH_3_1"},
                          8: {"x": 250, "y": 242.5, "desc_y": "V_2_3", "desc_x": "H_3_2", "desc_xi": "VH_3_2"},
                          9: {"x": 305, "y": 242.5, "desc_y": "V_3_3", "desc_x": "H_3_3", "desc_xi": "VH_3_3"},
                          }

    gauge_length = 6
    gauge_width = 2.6
    angle_gauge_b = np.radians(-45)

    # iterate over all .csv files with nodal simulation results
    nodal_result_filenames = sorted(extracted_nodal_data.glob('*bottom_nodal.csv'))

    # get current damage state
    [load_case, submodel, damage_state, instance, nodal] = nodal_result_filenames[0].stem.split('_')
    # get damage label
    damage_label = 0 if damage_state == 'pristine' else 1

    # get current data source
    source = extracted_nodal_data.parents[1].name

    # construct filename for resulting csv file
    filename = f"VSSG_submodel_bottom_{damage_state}.csv"
    # create key structure of resulting file
    if damage_state == "pristine":
        result_keys = ['loadcase', 'damage_label', 'damage_state', 'source',
                       'V_1_1', 'V_2_1', 'V_3_1',
                       'V_1_2', 'V_2_2', 'V_3_2',
                       'V_1_3', 'V_2_3', 'V_3_3',
                       'H_1_1', 'H_1_2', 'H_1_3',
                       'H_2_1', 'H_2_2', 'H_2_3',
                       'H_3_1', 'H_3_2', 'H_3_3',
                       'VH_1_1', 'VH_1_2', 'VH_1_3',
                       'VH_2_1', 'VH_2_2', 'VH_2_3',
                       'VH_3_1', 'VH_3_2', 'VH_3_3']
    else:
        result_keys = ['loadcase', 'damage_label', 'damage_state', 'x', 'y', 'r', 'source',
                       'V_1_1', 'V_2_1', 'V_3_1',
                       'V_1_2', 'V_2_2', 'V_3_2',
                       'V_1_3', 'V_2_3', 'V_3_3',
                       'H_1_1', 'H_1_2', 'H_1_3',
                       'H_2_1', 'H_2_2', 'H_2_3',
                       'H_3_1', 'H_3_2', 'H_3_3',
                       'VH_1_1', 'VH_1_2', 'VH_1_3',
                       'VH_2_1', 'VH_2_2', 'VH_2_3',
                       'VH_3_1', 'VH_3_2', 'VH_3_3']

    with open(extracted_nodal_data.parent / filename, 'w', newline='') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=result_keys)
        # write header
        csvwriter.writeheader()

        # go through results file of each loadcase
        for nodal_results in tqdm(nodal_result_filenames, "interpolation of nodal result files: "):
            # dict to store row data of results file
            results_file_row = {}
            # add damage label to row
            results_file_row['damage_label'] = damage_label
            # add data source
            results_file_row['source'] = source
            # add damage state
            results_file_row['damage_state'] = damage_state
            # filter out loadcase number from filename
            results_file_row['loadcase'] = nodal_results.name.split('_')[0]
            if damage_state != "pristine":
                # get damage location from basis file
                df_basis_damage_locations = pd.read_csv('../submodels/Basis_damage_locations.csv', sep=',')
                df_basis_damage_locations.set_index('name', inplace=True)
                results_file_row['x'] = df_basis_damage_locations.loc[damage_state, 'x']
                results_file_row['y'] = df_basis_damage_locations.loc[damage_state, 'y']
                results_file_row['r'] = df_basis_damage_locations.loc[damage_state, 'radius']

            # read csv file (one loadcase) into pandas dataframe
            df_nodal_results = pd.read_csv(nodal_results, sep=',')
            node_coords = np.stack((df_nodal_results['x'], df_nodal_results['y']), axis=1)

            # determine interpolation points for each sensor
            # (depending on amount and sensor orientation reg. glob. coords)
            for sensor_idx, sensor in sensor_information.items():
                # points at which to interpolate data -> ndarray shape (m,D)
                inter_coords_a_x = np.array([[sensor['x'] - gauge_length/2, sensor['y'] - gauge_width/2],
                                           [sensor['x'] + gauge_length/2, sensor['y'] - gauge_width/2],
                                           [sensor['x'] + gauge_length/2, sensor['y'] + gauge_width/2],
                                           [sensor['x'] - gauge_length/2, sensor['y'] + gauge_width/2]])

                inter_coords_c_y = np.array([[sensor['x'] - gauge_width/2, sensor['y'] - gauge_length/2],
                                           [sensor['x'] + gauge_width/2, sensor['y'] - gauge_length/2],
                                           [sensor['x'] + gauge_width/2, sensor['y'] + gauge_length/2],
                                           [sensor['x'] - gauge_width/2, sensor['y'] + gauge_length/2]])
                # This calculation assumes a strain gauge with 0/45/90 setup!
                inter_coords_b_xi = np.array([[sensor['x'] + np.sqrt(2)/2 * (- gauge_length/2 + gauge_width/2),
                                               sensor['y'] + np.sqrt(2)/2 * (- gauge_length/2 - gauge_width/2)],
                                              [sensor['x'] + np.sqrt(2)/2 * (+ gauge_length/2 + gauge_width/2),
                                               sensor['y'] + np.sqrt(2)/2 * (+ gauge_length/2 - gauge_width/2)],
                                              [sensor['x'] + np.sqrt(2)/2 * (+ gauge_length/2 - gauge_width/2),
                                               sensor['y'] + np.sqrt(2)/2 * (+ gauge_length/2 + gauge_width/2)],
                                              [sensor['x'] + np.sqrt(2)/2 * (- gauge_length/2 - gauge_width/2),
                                               sensor['y'] + np.sqrt(2)/2 * (- gauge_length/2 + gauge_width/2)]
                                              ])
                # average over interpolated points
                Exx_mean = scaling * np.mean(
                    griddata(node_coords, df_nodal_results['E_11'], inter_coords_a_x, method='linear'))
                Eyy_mean = scaling * np.mean(
                    griddata(node_coords, df_nodal_results['E_22'], inter_coords_c_y, method='linear'))
                Exy_mean = scaling * np.mean(
                    griddata(node_coords, df_nodal_results['E_12'], inter_coords_b_xi, method='linear'))
                # convert engineering shear strain to strain in 45 deg direction
                Eqq_mean = 1/2 * (Exx_mean + Eyy_mean) + 1/2 * (Exx_mean - Eyy_mean) * np.cos(2*angle_gauge_b) + 1/2 * Exy_mean * np.sin(2*angle_gauge_b)

                # write virtual strain value of each sensor to results row
                results_file_row[sensor['desc_x']] = Exx_mean
                results_file_row[sensor['desc_y']] = Eyy_mean
                results_file_row[sensor['desc_xi']] = Eqq_mean

            # write results directly to csv file
            csvwriter.writerow(results_file_row)
    print(f"CSV file saved to {extracted_nodal_data.parent / filename}")


def clean_up_dir(dir, suffix_to_remove=None, files_to_remove=None):
    """
    Removes files with specified suffix from directory.

    For the extraction of nodal information only the .odb file is needed.
    Deletes .msg / .sta / .sim / .prt / .inp / .com -> KEEP .obd / .dat
    .inp file of submodel and global model can be safely deleted since they are only copies of the
    respective files in the submodels and global models folder.

    :param dir: directory of submodel simulation results
    :param suffix_to_remove: -> list: list of extensions to remove
    :param files_to_remove: -> list: list of absolute paths of specific files to remove
    :return: None
    """

    if suffix_to_remove is None:
        suffix_to_remove = ['msg', 'sta', 'sim', 'prt', 'inp', 'com']

    if files_to_remove is None:
        files_to_remove = []

    for ext in suffix_to_remove:
        # gather paths of files with extensions that are to be removed
        files_to_remove.extend(sorted(dir.glob(f'*.{ext}')))

    # add global model files to removal list
    if len(sorted(dir.glob('Global*.odb'))) > 0:
        files_to_remove.extend(sorted(dir.glob('Global*.odb')))

    for file in files_to_remove:
        try:
            # delete file via pathlib function
            file.unlink()
        except FileNotFoundError:
            print(f"File: {file} not found")


if __name__ == '__main__':
    # pristine submodel
    #submodel_path = Path('D:/Masters_Thesis/work/models/FEM_model/submodels/submodel_pristine.inp')
    submodel_path = Path('D:/Masters_Thesis/work/models/FEM_model/submodels/submodel_DS0.inp')
    # target directory
    target_dir = Path('D:/Masters_Thesis/work/models/framework/input_data')
    global_lc = {'odb': Path('D:/Masters_Thesis/work/models/FEM_model/global_loadcases/Global_0.odb'),
                 'prt': Path('D:/Masters_Thesis/work/models/FEM_model/global_loadcases/Global_0.prt')}

    # call function
    #generate_stiffness_matrices(submodel_path, global_lc, target_dir)