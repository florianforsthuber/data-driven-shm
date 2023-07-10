"""Helper module to start and manage simulation of submodels
Manages simulation of submodels and the associated directory structure.
Detailed documentation is provided in the individual functions

General outline:
* take one submodel from /submodels
* create a folder in /simulation_results according to the damage state of this submodel
* for EACH global loadcase run the submodel analysis inside newly created directory
    * copy the respective global loadcase to results folder
    * copy the submodel to the results folder
    * run the submodel analysis
    * rename the submodel result file (.odb)
    * clean up all unnecessary files (keep submodel.odb and maybe .dat -> for EZT)
* move back to parent directory
* extract the node results for THIS submodel and all loadcases

Author: student k1256205@students.jku.at
Created: 22/06/2022
"""
import os
import shutil
from pathlib import Path
from numpy.random import default_rng
from tqdm import tqdm

# import general FEM functions by name for better readability
from FEM_functions import extract_nodal_results
from FEM_functions import virtual_strain_data
from FEM_functions import clean_up_dir

def run_submodel_analysis(submodel_path, global_loadcase_path, remove_excess_files=False):
    """
    Procedure takes the path of one submodel .inp file,
    creates a new folder (if it doesn't already exist) with the name of the
    damage id in the /simulation_results directory and
    copies the submodel .inp file to the new directory.
    The file is renamed according to the current global_loadcase.
    - Naming convention: {Loadcase Nr.}_{submodel}_{submodel_part}_{damage_state}_{results_format}
    The global loadcase .odb file is also copied into this folder to run the submodel analysis.

    This function needs to be called for every submodel / global model pairing.

    :param submodel_path: pathlib directory of submodel .inp file
    :param global_loadcase_path: dict with pathlib directory of global loadcase .odb file and .prt file
    :param remove_excess_files: set flag True to clean up directory after simulation is done
    :return: new_odb_filepath, odb_error_flag: path of .odb file with simulation results of
    damagae_state - loadcase pairing and error flag indicating, wether .odb_f file was created.
    """

    # get current directories
    p_cwd = Path.cwd()
    p_root = p_cwd.parent

    # check if directory is of type Path
    if not isinstance(submodel_path, Path):
        submodel_path = Path(submodel_path)

    # get damage state info from submodel .inp file
    submodel_filename = submodel_path.stem
    damage_state = submodel_filename.split('_')[-1]


    new_dir = p_root / 'simulation_results' / damage_state

    # create new directory in /simulation_results
    try:
        new_dir.mkdir()
    except FileExistsError:
        print(f"No new directory created. Directory {new_dir} already exists.")

    # get loadcase number
    global_odb_filename = str(global_loadcase_path['odb'].stem)
    loadcase_number = global_odb_filename.split('_')[-1]

    # rename and copy submodel .inp file to new directory
    # new: no renaming necessary -> nodal data focus after extraction
    new_submodel_name = f"{loadcase_number}_{submodel_filename}.inp"
    # if destination file already exists, it will be replaced
    shutil.copy(submodel_path, new_dir / new_submodel_name)

    # copy global loadcase .odb file to new directory
    global_odb_path = global_loadcase_path['odb']
    shutil.copy(global_odb_path, new_dir / global_odb_path.name)

    # copy global loadcase .prt file to new directory
    global_prt_path = global_loadcase_path['prt']
    shutil.copy(global_prt_path, new_dir / global_prt_path.name)

    # run analysis inside new directory
    try:
        os.chdir(new_dir)
        os.system(
            f"call abaqus job={new_submodel_name} globalmodel={global_loadcase_path['odb'].stem} cpus=6 interactive ask_delete=OFF")
    finally:
        os.chdir(p_cwd)

    if remove_excess_files:
        clean_up_dir(new_dir)

    new_odb_filepaths = sorted(new_dir.glob(f"{new_submodel_name.split('.')[0]}.odb*")) #TODO: NOT TESTED -> check if this works
    if ".odb_f" in new_odb_filepaths:
        odb_error_flag = True
        #return paths of odb_f and odb file for logging
        new_odb_filepath = new_odb_filepaths
    else:
        assert len(new_odb_filepaths) == 1 and new_odb_filepaths[0].is_file(), "new_odb_filepaths has more than one elements!"
        odb_error_flag = False
        new_odb_filepath = new_odb_filepaths[0]

    # return path of newly created odb file and simulation error indicator
    return new_odb_filepath, odb_error_flag


def sample_unique_loadcases(rng, global_lc_numbers, submodel_results, amount, PRINT_STATUS=False):
    """
    Randomly picks new loadcases which have not been simulated.
    There will be as many loadcases picked, as it is necessary to fullfill the
    requested amount of loadcases to simulate.

    :param rng: seeded random generator (numpy)
    :param loadcase_dir: directory of all available global loadcases
    :param submodel_results: directory of submodel results of a specific damage state. Does not need to exist!
    :param amount: requested total number of loadcases to simulate
    :return: list of global loadcase numbers which do not already exist in submodel results directory.
             returns empty list if requested amount of loadcases is already available in simulation results directory.
    """

    # get list of existing submodel loadcases
    if submodel_results.is_dir():
        submodel_loadcases = sorted(submodel_results.glob("*.odb"))
        submodel_lc_numbers = [loadcase_name.stem.split('_')[0] for loadcase_name in submodel_loadcases]
    else:
        # submodel simulation results directory does not exist yet
        submodel_lc_numbers = []

    # available loadcase numbers -> remove submodel_lc_numbers from global_lc_numbers
    available_loadcases = [lc for lc in global_lc_numbers if lc not in submodel_lc_numbers]

    # avoid picking from an empty list -> available loadcases secures this case
    assert len(available_loadcases) >= amount - len(submodel_lc_numbers), f"Only {len(available_loadcases)} loadcases available!"
    # pick new loadcases to fill up sim. results to amount_to_simulate
    new_selection = []
    while len(new_selection) < amount - len(submodel_lc_numbers): # change mmeaning of amount to total number -> fill up to it
        potential_pick = rng.choice(available_loadcases, replace=False)
        if potential_pick not in submodel_lc_numbers and potential_pick not in new_selection:
            new_selection.append(potential_pick)

    if PRINT_STATUS:
        if amount - len(submodel_lc_numbers) <= 0:
            print(f"--- STATUS: No new loadcases picked, requested amount already available")

    return new_selection


def handle_simulation(rng,
                      nr_loadcases_to_sim,
                      submodels_to_sim,
                      pristine=False,
                      instance="SUB-SKIN-BOTTOM-1",
                      step="Step-3_Load"):
    """
    handles the simulation procedure of the submodel - global loadcase pairs.

    :param rng: numpy random generator
    :param nr_loadcases_to_sim: total number of loadcases for one submodel -> see sample_unique_loadcases
    :param submodels_to_sim: dict -> range of submodels to simulate
    :param pristine: bool -> flag to distinguish between pristine and damaged submodels
    :param instance: str -> string specifying which instance to extract nodal results from
    :param step: str -> string specifying which simulation step holds results -> see extract_nodal_results
    :return:
    """

    # check if pristine or damaged submodels should be simulated
    if pristine:
        submodels_to_sim['start'] = 0
        submodels_to_sim['end'] = 0

    # go through each submodel -> include end number
    for submodel_idx in range(submodels_to_sim['start'], submodels_to_sim['end'] + 1):
        # directories of global loadcases
        loadcase_dir = Path.cwd().parent / 'global_loadcases'

        if pristine:
            # assume filenaming is correct, otherwise abort
            submodel_filepath = Path.cwd().parent / 'submodels' / f"submodel_pristine.inp"
            assert submodel_filepath.is_file(), f"File < submodel_pristine.inp > does not exist!"

            # path of simulation results -> needs not to exist necessarily! Will be created if it does not exist
            submodel_results = Path.cwd().parent / 'simulation_results' / f"pristine"
        else:
            # assume filenaming is correct, otherwise abort
            # TODO: change naming convention for future simulations -> bottom already to specific
            submodel_filepath = Path.cwd().parent / 'submodels' / f"submodel_DS{submodel_idx}.inp"
            assert submodel_filepath.is_file(), f"File < submodel_DS{submodel_idx}.inp > does not exist!"

            # path of simulation results -> needs not to exist necessarily!
            submodel_results = Path.cwd().parent / 'simulation_results' / f"DS{submodel_idx}"

        # get list of available global loadcase numbers
        global_loadcases = sorted(loadcase_dir.glob("Global_*.odb"))
        global_lc_numbers = [loadcase_name.stem.split('_')[-1] for loadcase_name in global_loadcases]
        # get specified amount of random loadcase numbers
        selected_loadcases = sample_unique_loadcases(rng, global_lc_numbers, submodel_results, nr_loadcases_to_sim)
        # get global loadcase filepaths
        if len(selected_loadcases) > 0:
            for loadcase_idx in tqdm(selected_loadcases, f"Simulate loadcases with {submodel_filepath.stem}"):
                global_loadcase_path = {'odb': loadcase_dir / f"Global_{loadcase_idx}.odb",
                                        'prt': loadcase_dir / f"Global_{loadcase_idx}.prt"}

                # run submodel analysis of one submodel - global loadcase pairing
                new_odb_filepath, odb_error_flag = run_submodel_analysis(submodel_filepath,
                                                                         global_loadcase_path,
                                                                         remove_excess_files=True)
                if not odb_error_flag:
                    # extract nodal results from the new simulation results -> .odb file
                    raw_data_dir = extract_nodal_results(new_odb_filepath,
                                                         instance=instance,
                                                         step=step)
                else:
                    # skip but log
                    # write .odb_f path to error log txt file in cwd()
                    # handle log file manually! -> delete once in a while because of append
                    with open("simulation_error_log.txt", "a") as log_file:
                        log_file.write(str(new_odb_filepath))

            # after simulation of all loadcases (one damage case) extract nodal results from .odb files

            # interpolate virtual strain data and write it to .csv file
            if raw_data_dir.is_dir():
                virtual_strain_data(raw_data_dir, strain_scaling="microstrain")

if __name__ == '__main__':

    rng = default_rng(seed=40)
    # control which files to simulate manually for now
    # damaged submodels
    #nr_loadcases_to_sim = 200
    ##nr_loadcases_to_sim = 2
    ##submodels_to_sim = {'start': 0, 'end': 53}

    ##handle_simulation(rng,
    ##                  nr_loadcases_to_sim,
    ##                  submodels_to_sim,
    ##                  pristine=False,
    ##                 instance="SUB-SKIN-BOTTOM-1",
    ##                  step="Step-3_Load")

    # test extract node data
    ##odb_filepath = Path("D:/Masters_Thesis/work/models/FEM_model/simulation_results/pristine/0_submodel_pristine.odb")
    instance = "SUB-SKIN-BOTTOM-1"
    step = "Step-3_Load"
    ##extract_nodal_results(odb_filepath, instance, step)

    loadcase_dir = Path("D:/Masters_Thesis/work/models/FEM_model/global_loadcases")
    loadcase_idx = 2

    submodel_filepath = Path("D:/Masters_Thesis/work/models/FEM_model/submodels/submodel_DS2.inp")
    global_loadcase_path = {'odb': loadcase_dir / f"Global_{loadcase_idx}.odb",
                            'prt': loadcase_dir / f"Global_{loadcase_idx}.prt"}

    # run submodel analysis of one submodel - global loadcase pairing
    new_odb_filepath, odb_error_flag = run_submodel_analysis(submodel_filepath,
                                                             global_loadcase_path,
                                                             remove_excess_files=True)
    # extract nodal results from the new simulation results -> .odb file
    raw_data_dir = extract_nodal_results(new_odb_filepath,
                                         instance=instance,
                                         step=step)
    #simulate_damaged_submodels(rng, nr_loadcases_to_sim, submodels_to_sim)

    # pristine submodel
    #nr_loadcases_to_sim = 1000
    #simulate_pristine_submodel(rng, nr_loadcases_to_sim)
