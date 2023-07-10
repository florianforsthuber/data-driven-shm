"""<Short description>
<Long description>
Author: student k1256205@students.jku.at
Created: 09/08/2022
"""
import pandas as pd
from scipy.sparse import coo_matrix
from scipy import linalg
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import polynomial_kernel
from pathlib import Path

from framework_helper_functions import get_node_maps
from framework_helper_functions import get_node_output
from framework_helper_functions import get_row_permutation

def static_condensation(input_data_DS_LC_dir, permutation, nr_master_DOFs, recompute=False):
    """
    1. COO representation of stiffness matrix is read.
    2. Stiffness matrix is reordered with permutation via advanced indexing.
    3. Submatrices are sliced.
    4. Reduced stiffness matrix is calculated and stored
    :param COO_matrix_path:
    :param permutation:
    :param nr_master_DOFs:
    :return: reduced stiffness matrix K_red
    """

    COO_matrix_path = sorted(input_data_DS_LC_dir.parent.glob('*COO.mtx'))[0]
    assert COO_matrix_path.is_file()

    K_red_file = COO_matrix_path.with_name(f"{COO_matrix_path.stem}_K_red.pkl")
    # check if results have already been dumped to pickle
    if not K_red_file.is_file() or recompute:
        print(f"Starting static condensation...")
        # 1. convert sparse COO matrix into full matrix
        df_COO_matrix = pd.read_csv(COO_matrix_path, delimiter=r"\s+", names=['row', 'col', 'data'])
        K_full = coo_matrix((df_COO_matrix.data, (df_COO_matrix.row, df_COO_matrix.col))).toarray()
        # eliminate 0 row and column -> indexing discrapency between abaqus and scipy
        K_full = K_full[1:, 1:]
        # 2. order rows of full matrix according to permutation (-> master DOFs)
        K_ordered = K_full[permutation]
        # order columns of ordered matrix according to permutation (-> master DOFs)
        K_ordered = K_ordered[:, permutation]

        # 3. build submatrices according to nr_master_DOFs
        K_mm = K_ordered[0:nr_master_DOFs, 0:nr_master_DOFs]
        K_ss = K_ordered[nr_master_DOFs:, nr_master_DOFs:]
        K_ms = K_ordered[0:nr_master_DOFs, nr_master_DOFs:]
        K_sm = K_ordered[nr_master_DOFs:, 0:nr_master_DOFs]

        # 4. calculate K_red with submatrices
        K_ss_inv = linalg.inv(K_ss)
        K_red = K_mm - K_ms @ K_ss_inv @ K_sm

        print(f"Static condensation COMPLETED")
        # dump K_red to pickle
        with open(K_red_file, 'wb') as file:
            pickle.dump(K_red, file)
    else:
        # files already exist -> just load data
        with open(K_red_file, 'rb') as file:
            K_red = pickle.load(file)
        print(f"Reduced stiffness matrix K_red <{K_red_file.name}> loaded.")

    return K_red

def eigenvector_decomposition(K_red):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric)
    or a real symmetric matrix.
    Returns two objects, a 1-D array containing the eigenvalues of a,
    and a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).
    The eigenvectors are written in ascending order,
    the column v[:, i] is the normalized (unit length) eigenvector
    corresponding to the eigenvalue w[i].
    :param K_red:
    :return: eigenvector matrix
    """

    # ! eigh() assumes a symmetric matrix !
    print(f"Starting eigenvector decomposition...")
    eigenvalues, eigenvectors = np.linalg.eigh(K_red)
    print(f"Eigenvector decomposition COMPLETED")
    return eigenvectors

def assemble_objective_loadcase(df_node_output, global_map: dict):
    """
    Turn node output into a column vector,
    objective loadcase is definied by the displacements of
    the master nodes and their respective DOFs

    np.isnan returns a boolean/logical array which has the value True everywhere that x is not-a-number.
    Since we want the opposite, we use the logical-not operator ~ to get an array with Trues everywhere
    that x is a valid number.
    Lastly, we use this logical array to index into the original array x.
    https://stackoverflow.com/questions/11620914/how-do-i-remove-nan-values-from-a-numpy-array
    :param df_node_output:
    :return: - numpy array containing displacements of the master DOFs, shape (nr_master_DOFs,),
    - list of dicts with mapping of every objective loadcase DOF ({'instance', 'local_nr', 'DOF1', 'DOF2'})
    """

    print(f"Starting objective loadcase assembly...")
    objective_lc_map = []
    for node in df_node_output['global_nr'].to_numpy():
        if global_map[node]['instance'] == 'SUB-CORE-1':
            for DOF in range(1, 4):
                row = {'instance': global_map[node]['instance'],
                       'local_nr': global_map[node]['local'],
                       'DOF1': DOF, 'DOF2': DOF}
                objective_lc_map.append(row)
        else:  # instance is either SUB-SKIN-TOP-1 or SUB-SKIN-BOTTOM-1
            for DOF in range(1, 7):
                row = {'instance': global_map[node]['instance'],
                       'local_nr': global_map[node]['local'],
                       'DOF1': DOF, 'DOF2': DOF}
                objective_lc_map.append(row)

    objective_loadcase = df_node_output.drop('global_nr', axis=1).to_numpy().flatten()
    print(f"Objective loadcase assembly COMPLETED")
    return objective_loadcase[~np.isnan(objective_loadcase)], objective_lc_map


def eigenvector_selection(eigenvectors, objective_loadcase, kernel=None, similarity_threshold=1.0e-2, nr_to_select=None):
    """
    Implementation of eigenvector selection. Sklearn pairwise metrics are assumed.
    The implementatioin of the original Framework paper is also available if kernel=None is passed.
    :param eigenvectors:
    :param objective_loadcase:
    :param kernel: pass sklearn_object directly!
    :param similarity_threshold:
    :param nr_to_select: number of top eigenvectors to pick, if not none it will overrule similarity_threshold
    :return: df_selected_eigv: Dataframe containing the selected eigenvectors (nr_master_DOFs, nr_selected)
    df_eigv_score_sorted: Dataframe containing the individual similarity scores of the eigenvectors (nr_master_DOFs, 2)
    """
    # find eigenvectors which best represent the objective loadcase in this space
    # reduce the number of eigenvectors to avoid excess simulations
    # consider groups of eigenvectors?
    # allow multiple methods

    objective_loadcase = objective_loadcase.reshape(-1, 1)
    eigenvector_similarity = []
    if kernel:
        assert kernel.__module__ == 'sklearn.metrics.pairwise'
        # use given sklearn kernel
        # expand objective loadcase from shape (n_features,) to shape (n_features, n_samples)

        for idx, eigenvector in enumerate(eigenvectors.transpose()):
            eigenvector = eigenvector.reshape(-1, 1)
            # kernels expect shape (n_samples_X, n_features) and return (n_samples_X, n_samples_Y)
            if kernel.__name__ == 'polynomial_kernel':
                similarity = kernel(eigenvector.transpose(), objective_loadcase.transpose(), degree=2, coef0=0)
            else:
                similarity = abs(kernel(eigenvector.transpose(), objective_loadcase.transpose()))
            eigenvector_similarity.append({'eigv_idx': idx, 'score': similarity.flatten()[0]})
    else:
        # do linear regression and COD
        for idx, eigenvector in enumerate(eigenvectors.transpose()):
            # shape of eigenvector? -> (1155,)
            # linear regression fit takes shape (1155,1) -> 1 coef_ else matrix
            eigenvector = eigenvector.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True)
            model.fit(eigenvector, objective_loadcase)
            COD = abs(model.score(eigenvector, objective_loadcase))
            eigenvector_similarity.append({'eigv_idx': idx, 'score': COD})

    df_eigenvector_similarity = pd.DataFrame(eigenvector_similarity)
    df_eigv_score_sorted = df_eigenvector_similarity.sort_values(by=['score'], ascending=False)
    if nr_to_select is None:
        selected_idx = df_eigv_score_sorted.loc[df_eigv_score_sorted['score'] > similarity_threshold, 'eigv_idx']
    else:
        selected_idx = df_eigv_score_sorted['eigv_idx'][:nr_to_select]
    # store selected eigenvectors in dataframe -> column: original eigenvector index
    df_selected_eigv = pd.DataFrame()
    selected_eigv = eigenvectors[:, selected_idx.to_list()]
    for idx, eigv in zip(selected_idx, selected_eigv.transpose()):
        df_selected_eigv[idx] = eigv
    # eigenvectors: df_selected_eigv.to_numpy()
    # selected_idx: df_selected_eigv.columns.to_list()
    return df_selected_eigv, df_eigv_score_sorted


def eigv_apporximation_parameter(df_selected_eigv, objective_loadcase, model=None):
    """
    Approximation of objective loadcase displacement vector through a (possibly) reduced number of eigenvectors.
    Essentially this is finds a parameter vector which scales each eigenvector as part of a linear combination w.r.t. to
    the euclidian distance to the OLC displacement vector.

    :param df_selected_eigv: Dataframe containing the selected eigenvectors (nr_master_DOFs, nr_selected)
    :param objective_loadcase: numpy array containing displacements of the master DOFs, shape (nr_master_DOFs,)
    :param model: optimization model to be used to find approximation parameters for selected eigenvectors w.r.t.
    objective loadcase (None: sklearn.LinearRegression, "lstsq": scipy.linalg.lstsq)
    :return: Dataframe with columns eigv_idx and value, holding the approximation parameter of that eigenvector
    """
    # find parameter vectors which approximate the objective loadcase best
    # eigenvectors: Number of N_B eigenvectors, shape: (nr_features, nr_selected)
    # possible models: - Linear regression / - plain scipy least squares / - kernel method
    #https://scicomp.stackexchange.com/questions/3245/choosing-subset-of-vectors-to-approximate-a-subspace
    # review: why does the amount of used eigenvectors influence the magnitude of the generated strain?
    #  why is the apporximation parameter not just bigger for less eigenvectors?

    df_x_coef = pd.DataFrame()
    df_x_coef['eigv_idx'] = df_selected_eigv.columns

    if model is None:
        # do linear regression
        model = LinearRegression(fit_intercept=False)
        # sklearn: X: shape (n_samples, n_features), y: shape (n_samples,) or (n_samples , n_targets)
        model.fit(df_selected_eigv.to_numpy(), objective_loadcase.reshape(-1, 1))
        df_x_coef['value'] = np.transpose(model.coef_.copy())

    elif model == "lstsq":
        # implement simple scipy.linalg.lstsq
        # Phi-matrix -> shape (n_master_DOFs, nr_EV_selected)
        # Um -> shape (nr_master_DOFs,)
        # this has the same result as LinearRegression model coeffs
        x, res, rank, s = linalg.lstsq(df_selected_eigv.to_numpy(), objective_loadcase)
        df_x_coef['value'] = x

    selected_eigv = df_selected_eigv.to_numpy()
    x_coef = df_x_coef['value'].to_numpy().reshape(-1, 1)
    approx_error = objective_loadcase.reshape(-1, 1) - selected_eigv @ x_coef
    distance = np.linalg.norm(approx_error)

    return df_x_coef, distance  # x_coef -> array of shape (n_features, ) or (n_targets, n_features)


def generate_samples(df_x_coef, rng, logfile,
                     objective_loadcase_nr,
                     eigv_strain_field_dir,
                     nr_to_generate=1,
                     distribution='uniform',
                     deviation=0.0,
                     custom_base_dir=None):
    """
    Short description: This function takes approximation parameters of the selected, best fitting eigenvectors,
    variates them according to a given distribution and returns the superposition of strain fields (all nodes!)
    produced by the individual eigenvector displacements. This is computationally slow and meant for development
    purposes, e.g. evaluation of the quality of the generated data.

    :param df_x_coef: DataFrame: approximation parameter for each selected eigenvector w.r.t selected objective loadcase
    :param rng: numpy random generator object
    :param distribution: type of distribution to be used to vary approx. parameters.
    Currently supported: 'uniform', 'normal'
    :param eigv_strain_field_dir: directory to nodal data of eigenvector displacement simulations
    :param nr_to_generate: amount of samples to generate through statistically variing the approximation parameter
    :param deviation: distribution deviation from mean 1. 'uniform': U(1-dev, 1+dev), 'normal': N(1, dev)
    :return: directory nodal_data of current damage_state and obj. loadcase -> holds subdirectories nrEV{X}
    """

    selected_eigv = []
    for idx in df_x_coef['eigv_idx']:
        path = list(eigv_strain_field_dir.glob(f"*_EV{idx}_*.csv"))
        assert len(path) == 1 and path[0].is_file()
        selected_eigv.append(path[0])

    # for every generated observation (loadcase) the approximation parameter is varied
    for i in tqdm(range(nr_to_generate), f"Generate {nr_to_generate} samples: "):
        df_varied_params = df_x_coef.copy()

        # statistically variate each approx_parameter
        if distribution == 'uniform':
            low = 1 - deviation
            high = 1 + deviation
            random_samples = rng.uniform(low, high, len(df_x_coef))
        elif distribution == 'normal':
            random_samples = rng.normal(1, deviation, len(df_x_coef))
        else:
            print(f"No variation of approximation parameters.")
            random_samples = np.ones_like(df_x_coef['value'])

        df_varied_params['value'] = df_varied_params['value'] * random_samples  # elementwise multiplication
        # superposition of nodal strainfield results * varied approx_parameters

        df_strain_field = strain_field_superposition(selected_eigv, df_varied_params)

        p_cwd = Path.cwd()
        if custom_base_dir is None:
            base_dir = p_cwd.parents[2] / 'data' / 'raw' / 'generated_full_field'
        else:
            base_dir = custom_base_dir

        damage_state = eigv_strain_field_dir.name
        file_name = f"{objective_loadcase_nr}-{i}_submodel_{damage_state}_nrEV{len(df_x_coef)}_bottom_nodal.csv"
        file_path = Path(f"{damage_state}", f"LC{objective_loadcase_nr}", 'nodal_data',
                         f"nrEV{len(df_x_coef)}", f"{file_name}")
        new_dir = base_dir.joinpath(file_path)
        new_dir.parent.mkdir(parents=True, exist_ok=True)

        df_strain_field.to_csv(new_dir, index=False)

    # log distribution settings
    logfile.log("nr_to_generate", nr_to_generate,
                heading=f"Generate samples based on objective loadcase {objective_loadcase_nr}")
    logfile.log("distribution", distribution,
                heading=f"Parameter variation distribution settings")
    if distribution == 'uniform':
        logfile.log("low", low)
        logfile.log("high", high)
    elif distribution == 'normal':
        # log distribution settings
        logfile.log("mean", 1)
        logfile.log("standard deviation", deviation)

    # return nodal_data dir
    return new_dir.parents[1], df_varied_params


def generate_samples_VSSG(df_x_coef, rng, logfile,
                     objective_loadcase_nr,
                     eigv_strain_field_dir,
                     nr_to_generate=1,
                     distribution='uniform',
                     deviation=0.0,
                     custom_base_dir=None):
    """
    Short description: This function takes approximation parameters of the selected, best fitting eigenvectors,
    variates them according to a given distribution and returns the superposition of VSSG strain data
    produced by the individual eigenvector displacements. This is computationally very inexpensive.

    :param df_x_coef: DataFrame: approximation parameter for each selected eigenvector w.r.t selected objective loadcase
    :param rng: numpy random generator object
    :param distribution: type of distribution to be used to vary approx. parameters.
    Currently supported: 'uniform', 'normal'
    :param eigv_strain_field_dir: directory to nodal data of eigenvector displacement simulations
    :param nr_to_generate: amount of samples to generate through statistically variing the approximation parameter
    :param deviation: distribution deviation from mean 1. 'uniform': U(1-dev, 1+dev), 'normal': N(1, dev)
    :return: directory nodal_data of current damage_state and obj. loadcase -> holds subdirectories nrEV{X}
    """

    # read eigenvector VSSG data into dataframe
    VSSG_EV_file = list(eigv_strain_field_dir.glob("VSSG_EVs*.csv"))[0]
    assert VSSG_EV_file.is_file()
    df_VSSG_EVs = pd.read_csv(VSSG_EV_file)

    # isolate the rows of requested eigenvectors
    df_isolated_EVs = df_VSSG_EVs[df_VSSG_EVs['eigv_idx'].isin(df_x_coef['eigv_idx'])]
    feature_names = [col for col in df_isolated_EVs.columns if 'V_' in col or 'H_' in col or 'VH_' in col]

    damage_label = df_isolated_EVs["damage_label"].values[0]
    damage_state = df_isolated_EVs["damage_state"].values[0]
    source = df_isolated_EVs["source"].values[0]
    if damage_state != "pristine":
        x = df_isolated_EVs["x"].values[0]
        y = df_isolated_EVs["y"].values[0]
        r = df_isolated_EVs["r"].values[0]

    df_x_coef = df_x_coef.sort_values("eigv_idx")
    df_isolated_EVs = df_isolated_EVs.sort_values("eigv_idx")
    assert np.array_equal(df_x_coef["eigv_idx"].values, df_isolated_EVs["eigv_idx"].values)

    # for every generated observation (loadcase) the approximation parameter is varied
    gen_samples_list = []
    for i in tqdm(range(nr_to_generate), f"Generate {nr_to_generate} samples: "):
        df_varied_params = df_x_coef.copy()
        df_EVs = df_isolated_EVs.copy()

        # statistically variate each approx_parameter
        if distribution == 'uniform':
            low = 1 - deviation
            high = 1 + deviation
            random_samples = rng.uniform(low, high, len(df_x_coef))
        elif distribution == 'normal':
            random_samples = rng.normal(1, deviation, len(df_x_coef))
        else:
            print(f"No variation of approximation parameters.")
            random_samples = np.ones_like(df_x_coef['value'])

        df_varied_params['value'] = df_varied_params['value'] * random_samples  # elementwise multiplication

        generated_VSSG_sample = np.transpose(df_varied_params['value'].values) @ df_EVs[feature_names]
        # add sampling index
        generated_VSSG_sample["sample_idx"] = i
        # store in list
        gen_samples_list.append(generated_VSSG_sample)

    # add labels to generated VSSG data
    df_generated_data = pd.DataFrame(gen_samples_list)
    df_generated_data = df_generated_data.astype({"sample_idx": "int"})
    df_generated_data["loadcase"] = objective_loadcase_nr
    df_generated_data["source"] = source
    df_generated_data["damage_state"] = damage_state
    df_generated_data["damage_label"] = damage_label
    if damage_state != "pristine":
        df_generated_data["x"] = x
        df_generated_data["y"] = y
        df_generated_data["r"] = r

    # log distribution settings
    logfile.log("nr_to_generate", nr_to_generate,
                heading=f"Generate samples based on objective loadcase {objective_loadcase_nr}")
    logfile.log("distribution", distribution,
                heading=f"Parameter variation distribution settings")
    if distribution == 'uniform':
        logfile.log("low", low)
        logfile.log("high", high)
    elif distribution == 'normal':
        # log distribution settings
        logfile.log("mean", 1)
        logfile.log("standard deviation", deviation)

    # save dataframe as csv file in data directory
    p_cwd = Path.cwd()
    if custom_base_dir is None:
       base_dir = p_cwd.parents[2] / 'data' / 'raw' / 'generated'
    else:
       base_dir = custom_base_dir

    filename = f"VSSG_submodel_bottom_{damage_state}_LC{objective_loadcase_nr}_nrEV{len(df_x_coef)}.csv"
    file_path = Path(f"{damage_state}", f"LC{objective_loadcase_nr}", f"{filename}")
    new_dir = base_dir.joinpath(file_path)
    new_dir.parent.mkdir(parents=True, exist_ok=True)

    df_generated_data.to_csv(new_dir, index=False)

    # return nodal_data dir
    return new_dir.parent, df_varied_params

def strain_field_superposition(selected_eigv_nodal_results, df_varied_params):
    """
    :param selected_eigv_nodal_results:
    :param df_varied_params:
    :return: dataFrame -> superposed starin field (reduced columns)
    """
    # isolate strains and displacements (old code) from .csv nodal file
    # collect strains caused by eigenvector displacements in axis=2 of numpy array
    # -> vectorized addition of all eigenvectors
    # eigv_nodal_results -> list of paths
    varied_data = []
    # loop over selected eigenvectors, those should be guaranteed in the list -> if not: new simulation required
    for nodal_result in selected_eigv_nodal_results:
        # get EV idx from filename
        EV_section = [idx for idx in nodal_result.stem.split('_') if 'EV' in idx][0]
        _, _, EV_idx = EV_section.partition('EV')
        assert EV_idx.isdigit()
        EV_idx = int(EV_idx)

        # only load strains and displacements from csv
        load_cols = ['nodeLabel', 'x', 'y', 'z',
                     'U_1', 'U_2', 'U_3', 'U_mag',
                     'UR_1', 'UR_2', 'UR_3', 'UR_mag',
                     'E_11', 'E_22', 'E_33', 'E_12']

        # read .csv data -> only read columns needed for later processing -> saves time
        df_nodal_results = pd.read_csv(nodal_result, sep=',', usecols=load_cols)

        isolate_cols = ['U_1', 'U_2', 'U_3', 'U_mag',
                        'UR_1', 'UR_2', 'UR_3', 'UR_mag',
                        'E_11', 'E_22', 'E_33', 'E_12']

        nodal_strain_disp = df_nodal_results[isolate_cols].to_numpy()
        # multiply nodal results of eigenvector displacement with corresponding approximation parameter
        varied_coef = df_varied_params[df_varied_params['eigv_idx'] == EV_idx]['value'].values[0]
        varied_data.append(nodal_strain_disp * varied_coef)
    # replace columns in original dataframe with varied and superposed columns
    nodal_superposition = np.sum(np.stack(varied_data), axis=0)

    # reformat numpy array to dataframe
    df_nodal_results[isolate_cols] = pd.DataFrame(nodal_superposition, columns=isolate_cols)

    return df_nodal_results






if __name__ == '__main__':
    from framework_helper_functions import create_eigv_disp_jobs
    from framework_helper_functions import simulate_eig_displacements
    from framework_helper_functions import get_strain_fields
    from framework_helper_functions import virtual_strain_data_eigv

    # 1. generate stiffness matrix -> path to directory holding stiffness matrices, etc.
    # DEBUG and TESTING
    COO_matrix_path = Path('D:/Masters_Thesis/work/models/framework/input_data/pristine/submodel_pristine_STIF_COO.mtx')
    MI_matrix_path = Path('D:/Masters_Thesis/work/models/framework/input_data/pristine/submodel_pristine_STIF_MI.mtx')
    dat_filepath = Path('D:/Masters_Thesis/work/models/framework/input_data/pristine/LC0/0_submodel_pristine_STIF.dat')
    template_path = Path('D:/Masters_Thesis/work/models/framework/input_data/pristine/LC0/0_submodel_pristine_EVtemplate.inp')

    global_map, _ = get_node_maps(dat_filepath)
    df_node_output = get_node_output(dat_filepath) # df_node_output should be equal to objective loadcase
    master_nodes = df_node_output['global_nr'].to_list()
    permutation, nr_master_DOFs = get_row_permutation(MI_matrix_path, master_nodes)

    K_red = static_condensation(COO_matrix_path, permutation, nr_master_DOFs)

    eigenvectors = eigenvector_decomposition(K_red)

    objective_loadcase, objective_lc_map = assemble_objective_loadcase(df_node_output, global_map)

    #df_selected_eigv, df_eigv_score_sorted
    df_selected_eigenvectors_LR, similarity_LR = eigenvector_selection(eigenvectors, objective_loadcase, kernel=None)
    df_selected_eigenvectors_cos, similarity_cosine = eigenvector_selection(eigenvectors, objective_loadcase, kernel=cosine_similarity)
    #selected_eigenvectors_poly, similarity_poly, selected_idx_poly = eigenvector_selection(eigenvectors, objective_loadcase, kernel=polynomial_kernel)

    job_dir = create_eigv_disp_jobs(df_selected_eigenvectors_LR, objective_lc_map, template_path)

    #sim_results_dir = simulate_eig_displacements(job_dir)
    sim_results_dir = Path('D:/Masters_Thesis/work/models/framework/eigenvector_simulation_results/pristine/LC0')
    results_dir = get_strain_fields(sim_results_dir)
    df_x_coef = eigv_apporximation_parameter(df_selected_eigenvectors_LR, objective_loadcase)

    rng = np.random.default_rng(0)
    eigv_strain_field_dir = Path('D:/Masters_Thesis/work/models/framework/eigenvector_strain_fields/pristine/LC0')

    custom_base_dir = Path('D:/Masters_Thesis/work/framework_parameter_test_data/variation/dev0-1')

    #gen_nodal_data_dir = generate_samples(df_x_coef, rng, eigv_strain_field_dir,
    #virtual_strain_data_eigv(gen_nodal_data_dir, strain_scaling="microstrain")
    pass
