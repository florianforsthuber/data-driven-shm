"""Helper functions for framework
Module holds helper functions for framework.
Functions might get integrated into framework architecture at a later stage.

Author: student k1256205@students.jku.at
Created: 08/08/2022
"""
import csv
import os
import shutil
import json
import filecmp
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


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


def get_submodel_inp_path(damage_state: str, objective_lc_nr: int):
    # get current working directory
    p_cwd = Path.cwd()

    # get relative submodel directory
    submodels_inp_dir = p_cwd.parents[1] / 'FEM_model' / 'submodels'
    assert submodels_inp_dir.is_dir() #smarter way to handle issues with relative path?

    # get relative global_lc directory
    global_lc_dir = p_cwd.parents[1] / 'FEM_model' / 'global_loadcases'
    assert global_lc_dir.is_dir()

    # get submodel .inp file of requested damage_state
    # filenames are standardized -> just construct filepath -> glob would be overkill
    submodel_inp_path = submodels_inp_dir / f"submodel_{damage_state}.inp"
    assert submodel_inp_path.is_file()

    # get global .obd file of requested loadcase
    global_lc_odb_path = global_lc_dir / f"Global_{objective_lc_nr}.odb"
    assert global_lc_odb_path.is_file()

    # get global .prt file of requested loadcase
    global_lc_prt_path = global_lc_dir / f"Global_{objective_lc_nr}.prt"
    assert global_lc_prt_path.is_file()

    return (submodel_inp_path, global_lc_odb_path, global_lc_prt_path)


def prepare_inp_files(file_paths: tuple, recompute=False):
    """
    - customizes input files for matrix and .dat file generation and
    eigenvector displacement template
    - matrix steps are added according to 'Generate_Matrix_inp_template.txt'
    - after matrix generation steps, step with submodel simulation and node output request
    is added
    - creates according standardized directories,
    base directory is /input_data -> every damage state gets subdirectory
    - Loadcase directory is already created at this stage, because stiffness matrix generation step
    also generates the .dat file holding the objective loadcase displacement values

    :param file_paths: tuple with (submodel_inp_path, global_lc_odb_path, global_lc_prt_path)
    :return: dst_filepath, template_path
    """
    (submodel_inp_path, global_lc_odb_path, global_lc_prt_path) = file_paths

    # get damage state
    damage_state = submodel_inp_path.stem.split('_')[-1]

    # get loadcase number
    loadcase_nr = global_lc_odb_path.stem.split('_')[-1]

    # create new directories
    new_dir = Path.cwd().parent / 'input_data' / f"{damage_state}" / 'eigenvector_displacements'
    try:
        new_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"No new directory created. Directory {new_dir} already exists.")

    new_LC_dir = Path.cwd().parent / 'input_data' / f"{damage_state}" / f"LC{loadcase_nr}"
    try:
        new_LC_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"No new directory created. Directory {new_dir} already exists.")


    # customize .inp file for matrix AND .dat file generation
    # copy submodel .inp file to target directory
    new_filename = f"{loadcase_nr}_{submodel_inp_path.stem}_STIF.inp"
    # path to new filename -> will be created by open method
    dst_filepath = new_LC_dir / new_filename

    # skip if .inp file already exists
    if not dst_filepath.is_file() or recompute:
        shutil.copy(submodel_inp_path, dst_filepath)

        # copy global loadcase .odb file to new directory
        shutil.copy(global_lc_odb_path, new_LC_dir / global_lc_odb_path.name)

        # copy global loadcase .prt file to new directory
        shutil.copy(global_lc_prt_path, new_LC_dir / global_lc_prt_path.name)

        # add Matrix Generate steps to .inp file
        with open(dst_filepath, 'r') as file:
            inp_file = file.read()

        # find keyword line where Node print keyword should be inserted
        # guaranteed unique because of .inp file definition
        sub_string = "*Output, history, variable=PRESELECT\n"
        # keyword line to replace substring -> adds a line below substring
        keyword_line_insertion = sub_string + "*NODE PRINT, NSET=SET-22\nU\n"
        inp_file = inp_file.replace(sub_string, keyword_line_insertion)

        # add Matrix generation steps BEFORE Step-3_Load to avoid 1e36 bug
        # add matrix generation steps from template file  to .inp file
        seperator = "** STEP: Step-3_Load"
        (inp_file_model_def, sep, inp_file_steps) = inp_file.partition(seperator)
        with open('Generate_Matrix_inp_template.txt', 'r') as template:
            inp_file = inp_file_model_def + template.read() + f"\n**\n{sep}" + inp_file_steps

        with open(dst_filepath, 'w') as new_inp_file:
            new_inp_file.write(inp_file)
    else:
        print(f"<{loadcase_nr}_{submodel_inp_path.stem}_STIF.inp> already exists. Creation skipped.")

    # prepare template for eigenvector displacement jobs -> write to new_dir
    # copy submodel .inp file to target directory
    EV_template_filename = f"{submodel_inp_path.stem}_EVtemplate.inp"
    template_path = new_dir.parent / EV_template_filename

    # skip if template file already exists
    if not template_path.is_file() or recompute:
        shutil.copy(submodel_inp_path, template_path)
        # remove submodel keyword and boundary lines
        with open(template_path, 'r') as file:
            template_file = file.read()

        # remove submodel keyword line
        old = "*Submodel, type=NODE, exteriorTolerance=0.05\nSet-22,"
        template_file = template_file.replace(old, '**')
        #template_file = model_def + steps

        # replace submodel boundary section with place holder
        sep_start = "** BOUNDARY CONDITIONS"
        idx_start = template_file.find(sep_start)
        sep_end = "** OUTPUT REQUESTS"
        idx_end = template_file.find(sep_end)
        place_holder = f"** BOUNDARY CONDITIONS\n" \
                       f"**\n" \
                       f"** Name: eigenvector displacements\n" \
                       f"*Boundary\n" \
                       f"** INSERT EV DISP BC\n" \
                       f"**\n"
        template_file = template_file[:idx_start] + place_holder + template_file[idx_end:]
        with open(template_path, 'w') as file:
            file.write(template_file)
    else:
        print(f"<{submodel_inp_path.stem}_EVtemplate.inp> already exists. Creation skipped.")

    return dst_filepath, template_path


def generate_matrices_and_olc(inp_file_path: Path, recompute=False):
    """
    1. submodel simulation of .inp file
    - function calls abaqus to execute matrix generation .inp file.
    - this job has to be executed as a submodel simulation, because after the
    generation of the stiffness matrix, a submodel simulation step is included
    to get the .dat file holding the displacements of the master nodes of this loadcase
    -> displacements of the objective loadcase
    2. directory clean up
    - unnecessary simulation files are removed
    - stiffness matrice .mtx files are renamed for clear identification and moved up one directory level, since
    they are identical for all loadcases (same damage_state!)

    :param inp_file_path: path to input file with matrix generation job and node output request
    :return: path of input_data/{damage_structure}
    """

    # check if .dat file corresponding to damage_state and load_case already exist
    # It's assumed that existence of .dat file indicates existence of .mtx files as well. Those should not be altered or deleted.
    # -> no need for matrix/dat generation simulation unless recomputation is requested
    if not inp_file_path.with_suffix('.dat').is_file() or recompute:
        # assert that global .odb and .prt files exist in directory of file_path
        global_lc_files = sorted(inp_file_path.parent.glob('Global_*'))
        assert len(global_lc_files) == 2

        # get current working directory
        p_cwd = Path.cwd()

        # call system abaqus -> generate matrices
        try:
            os.chdir(inp_file_path.parent)
            print(f"Starting matrix and .dat file generation...")
            os.system(
                f"call abaqus job={inp_file_path.name} globalmodel={global_lc_files[0].stem} cpus=6 interactive ask_delete=OFF")
            print(f"Generation of matrix and .dat file COMPLETED")
        finally:
            os.chdir(p_cwd)

        # delete miscellaneous files
        suffix_to_remove = ['com', 'msg', 'odb', 'prt', 'sim', 'sta']
        clean_up_dir(inp_file_path.parent, suffix_to_remove)

        # rename stiffness matrix to according to their format
        files = inp_file_path.parent.glob('*.mtx')
        for file in files:
            if "STIF1" in file.stem:
                # change STIF1 descriptor to MI -> Matrix-Input format
                rename_stif_files(file, old="STIF1", new="MI")
            elif "STIF2" in file.stem:
                # change STIF2 descriptor to COO -> Coordinate format
                rename_stif_files(file, old="STIF2", new="COO")
    else:
        print(f".dat file with OLC displacements already exists.")


    return inp_file_path.parent


def rename_stif_files(file: Path, old: str, new: str):
    """
    subroutine to handle renaming and moving stiffness matrix .mtx files.
    moves renamed file up one directory level
    :param file: file path of .mtx file
    :param old: old file ending to be replaced
    :param new: new file ending to replace old with
    :return: None
    """
    new_name = file.name.replace(old, new)
    # remove loadcase number
    new_name = new_name.split(sep='_', maxsplit=1)[-1]
    try:
        file.rename(file.parent.with_name(new_name))
    except FileExistsError:
        # check if files are identical
        print(f"File < {new_name} > already exists.")
        if filecmp.cmp(file.parent.with_name(new_name), file):
            # if yes remove current
            file.unlink()
        else:
            raise ValueError(f"Stiffness matrix file {new_name} has changed!")


def create_eigv_disp_jobs(df_selected_eigv, olc_node_map, input_data_DS_LC_dir, recompute=False):
    """
    create .inp files of the eigenvector displacement simulation jobs.
    write .inp file for every eigenvector. Eigenvector components are
    applied to master nodes as displacement boundary conditions.

    :param df_selected_eigv: DataFrame -> each column is a selected eigenvector, column label is original
    eigenvector label, shape: (nr_master_DOFs, nr_selected)
    :param olc_node_map: list of dicts -> instance, node and DOF information from assemble_loadcase()
    :param template_path: path of template .inp file
    :param recompute: Flag to indicate that .inp files should be newly created and overwrite the old ones
    :return: None
    """
    #eigenvector .inp files of the same structure (damage_state) are identical,
    # so only create .inp files if they do not exist already

    template_path = sorted(input_data_DS_LC_dir.parent.glob('*EVtemplate.inp'))[0]
    assert template_path.is_file()

    for eigv_idx in df_selected_eigv.columns.to_list():
        # copy template file and rename it according to selected eigenvector
        new_filename = template_path.name.replace("EVtemplate", f"EV{eigv_idx}")
        new_filepath = template_path.parent / 'eigenvector_displacements' / new_filename

        # check if file exists and if files should be newly created again
        if not new_filepath.is_file() or recompute:
            print(f"created new eigenvector .inp file <{new_filepath.name}>")

            shutil.copy(template_path, new_filepath)
            # construct strings of eigenvector displacements
            boundary_str = str()
            for idx, disp in enumerate(df_selected_eigv[eigv_idx]):  # should be like ndarray (1155,1)
                inst = olc_node_map[idx]['instance']
                local_nr = olc_node_map[idx]['local_nr']
                DOF1 = olc_node_map[idx]['DOF1']
                DOF2 = olc_node_map[idx]['DOF2']

                row = f"{inst}.{local_nr}, {DOF1}, {DOF2}, {disp}\n"
                boundary_str += row

            # replace place holder line with eigenvector displacements
            with open(new_filepath, 'r') as file:
                inp_file = file.read()
            place_holder = "** INSERT EV DISP BC"
            inp_before, _, inp_after = inp_file.partition(place_holder)
            # remove last line feed \n from boundary_str
            inp_file = inp_before + boundary_str[:-1] + inp_after

            with open(new_filepath, 'w') as file:
                file.write(inp_file)
            print(f"Eigenvector displacement .inp file <{new_filename}> created.")
        #else:
            #print(f"Eigenvector displacement .inp file <{new_filename}> already exists.")

    dir_inp_files = new_filepath.parent
    return dir_inp_files


def simulate_eig_displacements(job_dir: Path, target_dir=None, recompute=False):
    """
    Takes directory of eigenvector displacement .inp files, copies them to target_dir and
    invokes abaqus simulation in target_dir.
    Standard target_dir: /framework/eigenvector_simulation_results/{damage_state} -> set target_dir=None
    After simulation is completed, unnecessary files are removed from directory.

    :param job_dir: directory with eigenvector displacement .inp files of a certain damage_state.
    should be ..{damage_state}/eigenvector_displacements
    :param target_dir: directory where simulations are executed and where result files will be created.
    None -> standardized directory /framework/eigenvector_simulation_results/{damage_state} is used
    :return: target_dir
    """
    # get list of all .inp files to simulate
    inp_file_paths = sorted(job_dir.glob('*EV*.inp'))

    filename_comp = inp_file_paths[0].stem.split('_')

    # get damage_state
    damage_state = filename_comp[-2]

    p_cwd = Path.cwd()
    if target_dir is None:
        target_dir = p_cwd.parent / 'eigenvector_simulation_results' / f"{damage_state}"

    try:
        target_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"Directory {target_dir} already exists. Creation skipped.")

    for inp_file_path in inp_file_paths:
        # check if result files already exist
        # Assumption: if .odb exists, then .dat file exists
        odb_path = target_dir / inp_file_path.with_suffix('.odb').name
        if not odb_path.is_file() or recompute:
            # copy input file to target_dir
            shutil.copy(inp_file_path, target_dir / inp_file_path.name)

            # invoke abaqus simulation
            try:
                os.chdir(target_dir)
                os.system(f"call abaqus job={inp_file_path.name} cpus=6 interactive ask_delete=OFF")
            finally:
                os.chdir(p_cwd)
            print(f"Eigenvector displacement .inp file <{inp_file_path.name}> simulated.")
            # clean up directory -> only .odb files in target_dir needed
            clean_up_dir(target_dir)
        #else:
            #print(f"Eigenvector displacement result .odb file <{odb_path.name}> already exists.")

    return target_dir


def get_strain_fields(odb_dir: Path, target_dir=None, instance=None, recompute=False):
    """
    Extracts the nodal values of the .odb results in odb_dir via the ABAQUS SCRIPT extract_node_data.py.
    Only extracts nodal data of .odb results if the corresponding .csv file does not yet exist or recompute
    flag is set to True
    :param odb_dir: Directory where eigenvector displacement .odb files are stored
    :param target_dir: custom target directory where results are stored. Standard -> None
    :param instance: custom instance to pass to extract_node_data.py
    :return: newly created directory where results are stored
    """

    if target_dir is None:
        target_dir = Path.cwd().parent / 'eigenvector_strain_fields'

    if instance is None:
        instance = "SUB-SKIN-BOTTOM-1"

    p_cwd = Path.cwd()

    # get list of all .odb files
    odb_file_paths = sorted(odb_dir.glob('*EV*.odb'))

    filename_comp = odb_file_paths[0].stem.split('_')

    # get damage_state
    damage_state = filename_comp[-2]

    new_dir = target_dir / f"{damage_state}"
    try:
        new_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"Directory {new_dir} already exists. Creation skipped.")


    for odb_file_path in odb_file_paths:
        # check if nodal results have already been extracted to new_dir
        csv_name = f"{odb_file_path.stem}_bottom_nodal.csv"
        csv_path = new_dir / csv_name
        if not csv_path.is_file() or recompute:
            # change current working directory to abaqus scripts
            os.chdir(Path.cwd().parents[1] / 'FEM_model' / 'scripts')

            # define data to be dumped to json file
            extraction_settings = {'odb_filepath': str(odb_file_path),
                                   'target_dir': str(new_dir),
                                   'Step': "Step-3_Load",
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
                os.chdir(p_cwd)
            # review: move computation of eigenvector VSSG data for the current directory here
            print(f"Eigenvector nodal result .csv file <{csv_name}> extracted.")
        #else:
            #print(f"Eigenvector nodal result .csv file <{csv_name}> already exists.")

    return new_dir


def get_node_maps(dat_filepath: Path, recompute=False):
    """
    This function parses the .dat file resulting from the matrix creation & node output request (Set-22) simulation.
    It extracts the global to local node map and returns two representations in the form of dicitonaries.
    :param dat_filepath: path to *STIF.dat file
    :return: global_to_local_node_map -> dict{global_nr: dict{local_nr, instance}},
             local_to_global_node_map -> dict{instance: {local_nr: global_nr}}
    """

    global_pkl_file = dat_filepath.with_name(f"{dat_filepath.stem}_global_node_map.pkl")
    local_pkl_file = dat_filepath.with_name(f"{dat_filepath.stem}_local_node_map.pkl")
    # check if results have already been dumped to pickle
    if not (global_pkl_file.is_file() and local_pkl_file.is_file()) or recompute:
        section_indicator = "GLOBAL TO LOCAL NODE AND ELEMENT MAPS"
        end_section_indicator = "element"
        global_to_local_node_map = {}
        local_to_global_node_map = {}
        # open .dat file
        with open(dat_filepath, 'r') as dat_file:
            # maybe use readline() ? Check if it is slow
            data = dat_file.read()
        node_map_section = data.partition(section_indicator)[-1]
        node_map_section = node_map_section.partition(end_section_indicator)[0]
        for line in node_map_section.splitlines():
            if len(line.split()) == 3:
                [global_nr, local_nr, instance_name] = line.split()
                # get global to local map
                global_to_local_node_map[int(global_nr)] = {'local': int(local_nr), 'instance': instance_name}
                # get local to global map
                # local nodes are ambiguous! also instance name required!
                if instance_name not in local_to_global_node_map.keys():
                    local_to_global_node_map[instance_name] = {int(local_nr): int(global_nr)}
                else:
                    local_to_global_node_map[instance_name].update({int(local_nr): int(global_nr)})
        # dump node maps to pickle
        with open(global_pkl_file, 'wb') as file:
            pickle.dump(global_to_local_node_map, file)
        with open(local_pkl_file, 'wb') as file:
            pickle.dump(local_to_global_node_map, file)
    else:
        # Files already exist -> just load data
        with open(global_pkl_file, 'rb') as file:
            global_to_local_node_map = pickle.load(file)
        print(f"Global node map <{global_pkl_file.name}> loaded.")

        with open(local_pkl_file, 'rb') as file:
            local_to_global_node_map = pickle.load(file)
        print(f"Local node map <{local_pkl_file.name}> loaded.")

    # probably better to use two separate dicts, because of multiple assignment of local nodes
    return global_to_local_node_map, local_to_global_node_map


def get_node_output(dat_filepath, recompute=False):
    """
    This function parses the .dat file resulting from the matrix creation & node output request (Set-22) simulation.
    It extracts the node output data, so the displacement values at the master nodes (Set-22). These are the nodal
    displacements of the objective loadcase.
    :param dat_filepath: path to *STIF.dat file
    :return: dataframe representation of the node output written in .dat file
    """
    # node output -> objective loadcase IS loadcase DEPENDENT (values at least, nodes should be the same)
    # -> still save to LC dir level
    node_output_file = dat_filepath.with_name(f"{dat_filepath.stem}_node_output.pkl")
    # check if results have already been dumped to pickle
    if not node_output_file.is_file() or recompute:
        section_indicator = "N O D E   O U T P U T"
        second_sec_indicator = "NODE FOOT-"
        end_section_indicator = "MAXIMUM"
        # save data in list of dicts
        node_output = []
        with open(dat_filepath, 'r') as dat_file:
            data = dat_file.read()
        node_output_section = data.partition(section_indicator)[-1]
        node_output_section = node_output_section.partition(second_sec_indicator)[-1]
        node_output_section = node_output_section.partition(end_section_indicator)[0]
        for line in node_output_section.splitlines():
            if len(line.split()) == 4:
                [global_nr, U1, U2, U3] = line.split()
                node_output.append({"global_nr": int(global_nr), "U1": float(U1), "U2": float(U2), "U3": float(U3)})
            elif len(line.split()) == 7:
                [global_nr, U1, U2, U3, UR1, UR2, UR3] = line.split()
                node_output.append({"global_nr": int(global_nr), "U1": float(U1), "U2": float(U2), "U3": float(U3),
                                    "UR1": float(UR1), "UR2": float(UR2), "UR3": float(UR3)})
        # dump node maps to pickle
        with open(node_output_file, 'wb') as file:
            pickle.dump(node_output, file)
    else:
        # files already exist -> just load data
        with open(node_output_file, 'rb') as file:
            node_output = pickle.load(file)
        print(f"Node output <{node_output_file.name}> loaded.")

    return pd.DataFrame(node_output)


def get_row_permutation(input_data_DS_LC_dir, df_node_output, recompute=False):
    """
    Reads Matrix-Input formatted sparse stiffness matrix, iterates over each line.
    If a diagonal element is found, which is also contained in set of master nodes (Set-22),
    then its row (with respect to only the diag elements) is added to the master list. If a diagonal element
    is not part of Set-22 its respective row index is added to th slave list.
    Both lists are concatenated after the iteration is complete.

    :param input_data_DS_LC_dir: directory of current damage_state load_case combination
    :param df_node_output: dataframe -> holding displacements U of master nodes (nodes of Set-22) (OLC displacements)
    :return: list of int -> row permutation for ordering of full matrix from COO-format,
             number of master DOFs found for substructuring of matrix
    """
    # permutation is damage_structure dependent!

    row_idx = 0
    permutation_master = []
    permutation_slave = []

    master_nodes = df_node_output['global_nr'].to_list()

    # get path of stiffness matrix file in Matrix-Input format
    MI_matrix_path = sorted(input_data_DS_LC_dir.parent.glob('*MI.mtx'))
    assert len(MI_matrix_path) == 1 and MI_matrix_path[0].is_file()
    MI_matrix_path = MI_matrix_path[0]

    permutation_file = input_data_DS_LC_dir / f"{MI_matrix_path.stem}_permutation.pkl"
    # check if results have already been dumped to pickle
    if not permutation_file.is_file() or recompute:
        print(f"Starting get_row_permutation()...")
        with open(MI_matrix_path, newline='') as MI_matrix_csv:
            csv_reader = csv.reader(MI_matrix_csv)
            for row in tqdm(csv_reader, "Iterate rows of Matrix-Input stiffness matrix: "):
                [row_node, row_DOF, col_node, col_DOF, data] = row
                row_node = int(row_node)
                row_DOF = int(row_DOF)
                col_node = int(col_node)
                col_DOF = int(col_DOF)
                # find diagonal element
                if row_node == col_node and row_DOF == col_DOF:
                    if row_node in master_nodes:
                        permutation_master.append(row_idx)
                        row_idx += 1
                    else:
                        permutation_slave.append(row_idx)
                        row_idx += 1
        nr_master_DOFs = len(permutation_master)
        # combine master and slave permutation to one list
        permutation = [*permutation_master, *permutation_slave]
        print(f"get_row_permutation() COMPLETED")
        # dump permutation to pickle
        with open(permutation_file, 'wb') as file:
            pickle.dump((permutation, nr_master_DOFs), file) #DEBUG: check if dumping tuple like this works
    else:
        # files already exist -> just load data
        with open(permutation_file, 'rb') as file:
            permutation, nr_master_DOFs = pickle.load(file)
        print(f"Permutation of stiffness matrix <{permutation_file.name}> loaded.")

    return permutation, nr_master_DOFs


def virtual_strain_data_eigv(extracted_nodal_data, strain_scaling="microstrain"):
    """
    Interpolation of virtual strain data at given sensor positions in bottom instance of the submodel.
    Assumption: 0/45/90 strain gauge
    :param extracted_nodal_data: Directory which stores .csv files with extracted nodal results for one submodel type
    :param strain_scaling: str -> flag controlling the scaling
    :return: None
    """
    #TODO: seperate generated samples from differentt nrEVs

    # interpolate virtual strain at given sensor positions.

    # set microstrain scale
    if strain_scaling == "microstrain":
        scaling = 10**6
    else:
        scaling = 1

    # TODO: store all this in json file?
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

    # check if directory is of type Path
    if not isinstance(extracted_nodal_data, Path):
        extracted_nodal_data = Path(extracted_nodal_data)

    # find all nrEV subdirectories in provided nodal_data directory
    subdir_list = [dir for dir in extracted_nodal_data.iterdir() if dir.is_dir()]

    for dir in subdir_list:
        # iterate over all .csv files with nodal simulation results
        nodal_result_filenames = sorted(dir.glob('*.csv'))

        # get current damage state
        [load_case, submodel, damage_state, nrEV, bottom, nodal] = nodal_result_filenames[0].stem.split('_')
        # get damage label
        damage_label = 0 if damage_state == 'pristine' else 1

        # get current data source
        source = extracted_nodal_data.parents[2].name
        obj_load_case = extracted_nodal_data.parents[0].name

        # construct filename for resulting csv file
        filename = f"VSSG_submodel_bottom_{damage_state}_{obj_load_case}_{nrEV}.csv"
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

        # TODO: in later stage: automate the naming of sensor directions to reduce hardcoding

        with open(extracted_nodal_data.parent / filename, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=result_keys)
            # write header
            csvwriter.writeheader()

            # go through results file of each loadcase
            for nodal_results in tqdm(nodal_result_filenames, "Sensor interpolation of nodal result files: "):
                # dict to store row data of results file
                results_file_row = {}
                # add damage label to row
                results_file_row['damage_label'] = damage_label
                # add data source and get loadcase number
                results_file_row['source'] = source
                results_file_row['loadcase'] = nodal_results.name.split('_')[0]
                # add damage state
                results_file_row['damage_state'] = damage_state
                # filter out loadcase number from filename
                if damage_state != "pristine":
                    # get damage location from basis file
                    #df_basis_damage_locations = pd.read_csv('../submodels/Basis_damage_locations.csv', sep=',')
                    damage_loc_file_path = Path.cwd().parents[1].joinpath('FEM_model/submodels/Basis_damage_locations.csv')
                    df_basis_damage_locations = pd.read_csv(damage_loc_file_path, sep=',')
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

    return extracted_nodal_data.parent

def virtual_strain_data_singleEVs(extracted_nodal_data, strain_scaling="microstrain"):
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

    damage_state = extracted_nodal_data.stem
    # construct filename for resulting csv file
    filename = f"VSSG_EVs_submodel_bottom_{damage_state}.csv"

    check_path = extracted_nodal_data / filename
    if check_path.is_file():
        file_exists = True
        if pd.read_csv(check_path).shape[0] != len(nodal_result_filenames):
            nrEV_mismatch = True
        else:
            nrEV_mismatch = False
    else:
        file_exists = False
    # do this only if file does not exist or number of csv files does not match df.shape[0]
    if not file_exists or nrEV_mismatch:
        # create key structure of resulting file
        if damage_state == "pristine":
            result_keys = ['eigv_idx', 'damage_label', 'damage_state', 'source',
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
            result_keys = ['eigv_idx', 'damage_label', 'damage_state', 'x', 'y', 'r', 'source',
                           'V_1_1', 'V_2_1', 'V_3_1',
                           'V_1_2', 'V_2_2', 'V_3_2',
                           'V_1_3', 'V_2_3', 'V_3_3',
                           'H_1_1', 'H_1_2', 'H_1_3',
                           'H_2_1', 'H_2_2', 'H_2_3',
                           'H_3_1', 'H_3_2', 'H_3_3',
                           'VH_1_1', 'VH_1_2', 'VH_1_3',
                           'VH_2_1', 'VH_2_2', 'VH_2_3',
                           'VH_3_1', 'VH_3_2', 'VH_3_3']

        # TODO: in later stage: automate the naming of sensor directions to reduce hardcoding

        with open(extracted_nodal_data/ filename, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=result_keys)
            # write header
            csvwriter.writeheader()

            # go through results file of each loadcase
            for nodal_results in tqdm(nodal_result_filenames, "VSSG extraction of eigenvector strain: "):
                [submodel, damage_state, EV, instance, nodal] = nodal_results.stem.split('_')
                EV_index = int(EV.split("EV")[1])
                damage_label = 0 if damage_state == 'pristine' else 1

                # dict to store row data of results file
                results_file_row = {}
                # add eigenvector index to row
                results_file_row['eigv_idx'] = EV_index
                # add damage label to row
                results_file_row['damage_label'] = damage_label
                # add data source and get loadcase number
                results_file_row['source'] = "generated"
                # add damage state
                results_file_row['damage_state'] = damage_state
                # filter out loadcase number from filename
                if damage_state != "pristine":
                    # get damage location from basis file
                    # df_basis_damage_locations = pd.read_csv('../submodels/Basis_damage_locations.csv', sep=',')
                    damage_loc_file_path = Path.cwd().parents[1].joinpath(
                        'FEM_model/submodels/Basis_damage_locations.csv')
                    df_basis_damage_locations = pd.read_csv(damage_loc_file_path, sep=',')
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
        print(f"CSV file saved to {extracted_nodal_data / filename}")

if __name__ == '__main__':
    pass