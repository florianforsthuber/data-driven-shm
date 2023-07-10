"""Main file for framework execution
Author: student k1256205@students.jku.at
Created: 11/08/2022
"""
from pathlib import Path
import numpy as np
from datetime import datetime
from datetime import timedelta
import time
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import polynomial_kernel

from framework_helper_functions import get_submodel_inp_path
from framework_helper_functions import prepare_inp_files
from framework_helper_functions import generate_matrices_and_olc
from framework_helper_functions import get_node_maps
from framework_helper_functions import get_node_output
from framework_helper_functions import get_row_permutation

from framework_architecture import static_condensation
from framework_architecture import eigenvector_decomposition
from framework_architecture import assemble_objective_loadcase
from framework_architecture import eigenvector_selection
from framework_architecture import eigv_apporximation_parameter
from framework_architecture import generate_samples
from framework_architecture import generate_samples_VSSG

from framework_helper_functions import create_eigv_disp_jobs
from framework_helper_functions import simulate_eig_displacements
from framework_helper_functions import get_strain_fields
from framework_helper_functions import virtual_strain_data_eigv
from framework_helper_functions import virtual_strain_data_singleEVs

#from plotting_functions import plot_eigenvector_importance

class LogFile:
    def __init__(self):
        # start list of strings with current date, time
        self.logfile = [f"Date and Time:", f"{datetime.now()}\n"]
        self.start_time = time.time()

    def log(self, desc, data, heading=None):
        # append to list of strings -> row of log file
        if heading is not None:
            self.logfile.append(f"\n{heading}\n")

        self.logfile.append(f"{desc}: {data}\n")

    def save(self, filepath):
        # write total progrma runtime as last line
        total_sec = time.time() - self.start_time
        self.logfile.append(f"\nTotal process runtime: {timedelta(seconds=total_sec)}\n")
        # save logfile as .txt file to given dir
        with open(filepath, 'w') as file:
            for row in self.logfile:
                file.write(row)

def execute_framework(damage_state, objective_loadcase_nr,
                      logfile,
                      rng=None,
                      similarity_model=None,
                      similarity_threshold=1.0e-2,
                      nr_eigv_to_select=None,
                      approx_model=None,
                      distribution='uniform',
                      deviation=0.1,
                      nr_to_generate=1,
                      full_field_generation=False,
                      recompute=False,
                      custom_base_dir=None,
                      store_x_coef=False):

    # add initial settings to logfile
    logfile.log("similarity_model", similarity_model, heading=f"Initial Settings:")
    logfile.log("similarity_threshold", similarity_threshold)
    logfile.log("nr_eigv_to_select", nr_eigv_to_select)
    logfile.log("distribution", distribution)
    logfile.log("deviation", deviation)
    logfile.log("nr_to_generate", nr_to_generate)
    logfile.log("recompute", recompute)

    if rng is None:
        # define random generator
        rng = np.random.default_rng(0)

    # get file paths for requested structure
    file_paths = get_submodel_inp_path(damage_state=damage_state, objective_lc_nr=objective_loadcase_nr)

    # prepare .inp files and directory structure for stiffness matrix generation and .dat file
    matrix_inp_path, template_path = prepare_inp_files(file_paths, recompute=recompute)

    # initialize structure
    # 1. generate stiffness matrix
    input_data_DS_LC_dir = generate_matrices_and_olc(matrix_inp_path, recompute=recompute)

    # 2. get master nodes from abaqus .dat file
    dat_filepath = sorted(input_data_DS_LC_dir.glob('*.dat'))
    assert len(dat_filepath) == 1 and dat_filepath[0].is_file()
    dat_filepath = dat_filepath[0]

    global_map, _ = get_node_maps(dat_filepath, recompute=recompute)
    df_node_output = get_node_output(dat_filepath, recompute=recompute)

    # 3. get permutation of stiffness matrix elements for static condensation
    permutation, nr_master_DOFs = get_row_permutation(input_data_DS_LC_dir, df_node_output, recompute=recompute)


    # 4. perform static condensation
    K_red = static_condensation(input_data_DS_LC_dir, permutation, nr_master_DOFs, recompute=recompute)

    # 5. perform eigenvector decomposition
    eigenvectors = eigenvector_decomposition(K_red)

    # 6. assemble objective loadcase
    objective_loadcase, objective_lc_map = assemble_objective_loadcase(df_node_output, global_map)

    # 7. perform eigenvector selection -> kernel: None -> linear regression, cosine_similarity
    df_selected_eigenvectors, df_similarity = eigenvector_selection(eigenvectors, objective_loadcase,
                                                                    kernel=similarity_model,
                                                                    similarity_threshold=similarity_threshold,
                                                                    nr_to_select=nr_eigv_to_select)


    # 8. get approximation parameter for selected eigenvectors and objective loadcase
    df_x_coef, distance = eigv_apporximation_parameter(df_selected_eigenvectors, objective_loadcase, model=approx_model)

    # log results
    logfile.log("objective_loadcase_nr", objective_loadcase_nr,
                heading=f"Selected eigenvectors and similarity score w.r.t objective loadcase:")
    logfile.log("similarity_model", "LinearRegression" if similarity_model is None else similarity_model.__name__)
    logfile.log("similarity_threshold", similarity_threshold)
    logfile.log("nr_eigv_to_select", nr_eigv_to_select)
    logfile.log("nr_eigv_selected", len(df_x_coef))
    df_selected_eigv_logdata = df_similarity.merge(df_x_coef)
    logfile.log("approximation parameter model", approx_model)
    logfile.log(f"similarity score and approx. param. value\n", df_selected_eigv_logdata.to_string(index=False))
    logfile.log(f"euclid. distance of OLC and linear combination of EV * x_coef", distance)

    # 9. write .inp files for eigenvector displacement jobs
    job_dir = create_eigv_disp_jobs(df_selected_eigenvectors, objective_lc_map, input_data_DS_LC_dir, recompute=recompute)

    # 10. simulate selected eigenvector displacements
    sim_results_dir = simulate_eig_displacements(job_dir, recompute=recompute)

    # 11. extract nodal strain fields from eigenvector displacement simulation results
    results_dir = get_strain_fields(sim_results_dir, recompute=recompute)

    if full_field_generation:
        # 12. generate samples through statistical variation of eigenvector displacement fields
        gen_nodal_data_dir, df_varied_params = generate_samples(df_x_coef, rng, logfile,
                                                                objective_loadcase_nr=objective_loadcase_nr,
                                                                eigv_strain_field_dir=results_dir,
                                                                nr_to_generate=nr_to_generate,
                                                                distribution=distribution,
                                                                deviation=deviation,
                                                                custom_base_dir=custom_base_dir)
        # 13. extract virtual strain sensor data from nodal results of generated samples
        virtual_strain_data_eigv(gen_nodal_data_dir, strain_scaling="microstrain")
    else:
        # extract single eigenvector VSSG file only if file does not exist or number of csv files does not match df.shape[0]
        virtual_strain_data_singleEVs(results_dir, strain_scaling="microstrain")
        # eigenvector VSSG file should now exist in results dir

        # 12. generate samples through statistical variation of eigenvector VSSG data
        gen_nodal_data_dir, df_varied_params = generate_samples_VSSG(df_x_coef, rng, logfile,
                                                                     objective_loadcase_nr=objective_loadcase_nr,
                                                                     eigv_strain_field_dir=results_dir,
                                                                     nr_to_generate=nr_to_generate,
                                                                     distribution=distribution,
                                                                     deviation=deviation,
                                                                     custom_base_dir=custom_base_dir)

    # write logfile to target  directory of virtual_strain_data_eigv()
    logfile_path = gen_nodal_data_dir / f"logfile_{damage_state}_LC{objective_loadcase_nr}_nrEV{len(df_x_coef)}.txt"

    # save parameter vector df_x_coef
    if store_x_coef:
        x_coef_path = gen_nodal_data_dir / f"df_x_coef_{damage_state}_LC{objective_loadcase_nr}_nrEV{len(df_x_coef)}.csv"
        df_varied_params.to_csv(x_coef_path, index=False)

    logfile.save(logfile_path)


if __name__ == '__main__':
    # TODO: introduce load flag to simulation functions
    from sklearn.metrics.pairwise import cosine_similarity

    # define random generator
    rng = np.random.default_rng(14)
    logfile = LogFile()

    custom_base_dir = Path("D:/Masters_Thesis/work/data/raw/generated_small_u01")
    # Spread sample generation over loadcases
    damage_states = ["pristine"]
    objective_LCs = list(range(1000))#[3]

    for damage_state in damage_states:
        for loadcase in objective_LCs:
            logfile = LogFile()
            # execute framework
            execute_framework(damage_state, loadcase, logfile, rng=rng,
                              similarity_model=cosine_similarity,
                              similarity_threshold=1.0e-5,
                              nr_eigv_to_select=None,
                              approx_model="lstsq",
                              distribution='uniform',
                              deviation=0.1,
                              nr_to_generate=25,
                              full_field_generation=False,
                              recompute=False,
                              custom_base_dir=custom_base_dir,
                              store_x_coef=False)




