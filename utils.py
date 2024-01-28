import numpy as np
import matplotlib.pyplot as plt
import nilearn 
from nilearn import datasets
from nilearn import connectome
import nibabel as nib
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import seaborn as sns
from tqdm.notebook import tqdm
import multinetx as mx
from sklearn.model_selection import train_test_split
import bct 
from scipy import stats


def fmri_data_preparation(data_path, site_ids, age_range, proportional_threshold=0.3, verbose=False, save_name=None):
    data_path_ASD = data_path + "Autism_Data/"
    data_path_TC = data_path + "Control_Data/"
    autism_data = datasets.fetch_abide_pcp(data_dir=data_path_ASD, band_pass_filtering=True, legacy_format=False, pipeline='cpac', DX_GROUP=1, SITE_ID=site_ids, AGE_AT_SCAN=age_range)
    control_data = datasets.fetch_abide_pcp(data_dir=data_path_TC, band_pass_filtering=True, legacy_format=False, pipeline='cpac', DX_GROUP=2, SITE_ID=site_ids, AGE_AT_SCAN=age_range)
    if verbose:
        print(f"Number of Autism subjects: {len(autism_data.func_preproc)}")
        print(f"Number of Control subjects: {len(control_data.func_preproc)}")
    # we choose the msdl atlas
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas["maps"]
    labels = atlas["labels"]
    num_ROIs = len(labels)

    # Create a masker to extract time series
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize="zscore_sample")
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")

    pop_correlation_matrices = np.zeros((len(autism_data.func_preproc), num_ROIs, num_ROIs))
    for i in tqdm(range(len(autism_data.func_preproc))):

        subj = autism_data.func_preproc[i]
        img = nib.load(subj)
        masker.fit(img)
        time_series = masker.transform(img)
        pop_correlation_matrices[i, :, :] = correlation_measure.fit_transform([time_series])[0]
    
    control_pop_correlation_matrices = np.zeros((len(control_data.func_preproc), num_ROIs, num_ROIs))
    for i in tqdm(range(len(control_data.func_preproc))):

        subj = control_data.func_preproc[i]
        img = nib.load(subj)
        masker.fit(img)
        time_series = masker.transform(img)
        control_pop_correlation_matrices[i, :, :] = correlation_measure.fit_transform([time_series])[0]

    # set diagonal to 0 - no self-connectivity
    for i in range(pop_correlation_matrices.shape[0]):
       np.fill_diagonal(pop_correlation_matrices[i, :, :], 0)
    for i in range(control_pop_correlation_matrices.shape[0]):
       np.fill_diagonal(control_pop_correlation_matrices[i, :, :], 0)

    for i in range(pop_correlation_matrices.shape[0]):
        pop_correlation_matrices[i, :, :] = bct.weight_conversion(bct.threshold_proportional(pop_correlation_matrices[i, :, :], proportional_threshold, copy=True), wcm='normalize', copy=True)
    for i in range(pop_correlation_matrices.shape[0]):
        control_pop_correlation_matrices[i, :, :] = bct.weight_conversion(bct.threshold_proportional(control_pop_correlation_matrices[i, :, :], proportional_threshold, copy=True), wcm='normalize', copy=True)

    X = np.concatenate((pop_correlation_matrices, control_pop_correlation_matrices), axis=0)
    y = np.concatenate((np.ones(len(autism_data.func_preproc)), np.zeros(len(control_data.func_preproc))), axis=0)
    
    if save_name is not None:
        save_name_X = "X_" +save_name + ".npy"
        save_name_y = "y_" +save_name + ".npy"
        np.save(save_name_X, X)
        np.save(save_name_y, y)
    else:
        np.save("X_NYU.npy", X)
        np.save("y_NYU.npy", y)
    

def multiplex_pagerank(population_corr_matrices, num_ROIs, labels, inter_layer_edges_weight, interconnection_per_layer_pair, save_name_pagerank=None, save_name_supra_adj=None, check_plot=False, show_best_features=False):
    # build supra-adjacency matrix
    supra_adjacency_matrix = build_supra_adj_matrix(population_corr_matrices, num_ROIs, inter_layer_edges_weight, interconnection_per_layer_pair, save_name=save_name_supra_adj, check_plot=check_plot)

    # compute pagerank
    compute_multiplex_features(supra_adjacency_matrix, num_ROIs, labels, save_name=save_name_pagerank, show_best_features=show_best_features)


def independent_pagerank(population_corr_matrices, y, num_ROIs, labels, r_pagerank=0.85, save_name=None, show_best_features=False):
    not_converged = compute_independent_features(population_corr_matrices, num_ROIs, labels, r_pagerank=r_pagerank, save_name=save_name, show_best_features=show_best_features)  
    Z = np.load(save_name)
    Z = np.delete(Z, not_converged, axis=0)
    y = np.delete(y, not_converged, axis=0)
    return Z, y, not_converged

# function called by multiplex_pagerank
def build_supra_adj_matrix(population_corr_matrices, num_ROIs, inter_layer_edges_weight, interconnection_per_layer_pair, save_name=None, check_plot=False):
    num_subjects = population_corr_matrices.shape[0]

    list_of_layers = [mx.from_numpy_array(population_corr_matrices[i, :, :]) for i in range(num_subjects)]

    adj_block = mx.lil_matrix(np.zeros((num_ROIs*num_subjects,num_ROIs*num_subjects)))

    for i in tqdm(range(num_subjects), desc="Building supra-adjacency matrix", leave=False):
        for j in range(num_subjects):
            if j > i:  
                strength_i = np.sum(population_corr_matrices[i, :, :], axis=1)
                strength_j = np.sum(population_corr_matrices[j, :, :], axis=1)
                # add strengths to adjacency matrix
                sum_strengths = strength_i + strength_j
                # compute indices of nodes with the 5 highest strength
                sorted_indices = np.argsort(sum_strengths)[::-1][:interconnection_per_layer_pair]
                # check if there is a connection between layer i and layer j
                interconnection_matrix = np.zeros((num_ROIs))
                interconnection_matrix[sorted_indices] = 1
                adj_block[i*num_ROIs: (i+1)*num_ROIs, j*num_ROIs: (j+1)*num_ROIs] = np.diag(interconnection_matrix)

    adj_block += adj_block.T

    mg = mx.MultilayerGraph(list_of_layers=list_of_layers, inter_adjacency_matrix=adj_block)
    mg.set_edges_weights(inter_layer_edges_weight=inter_layer_edges_weight)

    if check_plot:
        plt.figure(figsize=(20,10))
        img = plt.imshow(mx.adjacency_matrix(mg,weight='weight').todense()[:3*num_ROIs, :3*num_ROIs],
                origin='upper',interpolation='nearest',cmap=plt.cm.Purples) #,vmin=0,vmax=1) TODO try different colormaps
        #plt.title('Supra adjacency matrix Autism Data first 3 subjects')

        cbar = plt.colorbar(img, cmap=plt.cm.plasma)
        # list of colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

        plt.show()

    if save_name is not None:
        np.save(save_name, mx.adjacency_matrix(mg,weight='weight').todense())
    
    supra_adj_matrix = mx.adjacency_matrix(mg,weight='weight').todense()
    
    return supra_adj_matrix

# function called by multiplex_pagerank
def compute_multiplex_features(supra_adjacency_matrix, num_ROIs, labels, r_pagerank=0.85, save_name=None, show_best_features=False):
    thresh = 0.00
    tol_inverse = 1e-3
    tol_leading_eigenval = 1e-1
    epsilon = 1e-10  # A small non-zero value
    r = r_pagerank
    num_subjects = int(supra_adjacency_matrix.shape[0]/num_ROIs)

    supra_adjacency_matrix = np.where(supra_adjacency_matrix<thresh, 0, supra_adjacency_matrix)
    strength = np.sum(supra_adjacency_matrix, axis=1)
    strength_matrix = np.diag(strength)
    strength_matrix_inv = np.where(np.isclose(strength_matrix,0), 0, 1/(strength_matrix+epsilon))

    # thresholding in case negative weights from numerical errors
    strength_matrix_inv = np.where(strength_matrix_inv<0+tol_inverse, 0, strength_matrix_inv)

    # equivalent as a tensor contraction between the supra-adjacency matrix and the inverse of the strength matrix
    transition_tensor = np.dot(strength_matrix_inv, supra_adjacency_matrix)

    # computing the eigenvalues random walk transition tensor (flattened)
    RW_transition_tensor = r * transition_tensor + (1-r)/transition_tensor.shape[0]
    
    eigvals, eigvecs = np.linalg.eigh(RW_transition_tensor)

    # Raise an error if the largest eigenvalue is not 1 (up to a tolerance) due to numerical errors
    
    if eigvals[-1] <= 1 - tol_leading_eigenval or eigvals[-1] >= 1 + tol_leading_eigenval :
        raise ValueError(f"Largest eigenvalue is not 1 but {eigvals[-1]} \n Leading eigenvector first entries {eigvecs[:10,-1]}.")
    leading_supra_eigenv = eigvecs[:, -1]

    if np.all(leading_supra_eigenv < 0): 
       leading_supra_eigenv = -leading_supra_eigenv
       #print("Negative entries of leading eigenvector")
    leading_eigen_tens = leading_supra_eigenv.reshape(num_subjects, num_ROIs)
    page_rank_centrality = np.mean(leading_eigen_tens, axis=0)

    # give the ordering of the nodes by page_rank_centrality
    page_rank_sorted_centrality = np.argsort(page_rank_centrality)[::-1]

    if show_best_features:
        # plot the 10 most central ROIs
        most_central_ROIs = [labels[page_rank_sorted_centrality[i]] for i in range(10)]
        print(most_central_ROIs)

    pagerank_score = leading_eigen_tens
    # saving pagerank scores to file
    if save_name is not None:
        np.save(save_name, pagerank_score)

# function called by independent_pagerank
def compute_independent_features(population_corr_matrices, num_ROIs, labels, r_pagerank=0.85, save_name=None, show_best_features=False):
    from networkx.exception import PowerIterationFailedConvergence
    
    num_subjects = population_corr_matrices.shape[0]

    graph_list = [mx.from_numpy_array(population_corr_matrices[i, :, :]) for i in range(num_subjects)]
    pagerank_scores = np.zeros((num_subjects, num_ROIs))

    not_converged = []
    for i in tqdm(range(num_subjects), desc="Independent centrality measures", leave=False): 
        try:
            pagerank_scores[i, :] = np.array(list(mx.pagerank(graph_list[i], max_iter=100000, tol=0.1, alpha=r_pagerank, personalization=None, nstart=None, weight='weight', dangling=None).values()))
        except PowerIterationFailedConvergence:
            pagerank_scores[i, :] = None
            not_converged.append(i)

    # remove not converged subjects
    print(f"Subjects for which Page Rank did not converge: {not_converged} \n Delete them manually from the dataset.")
    #pagerank_scores = np.delete(pagerank_scores, not_converged, axis=0)

    # mean of each node across subjects
    mean_pagerank_scores = np.mean(pagerank_scores, axis=0)

    if show_best_features:
        sorted_pagerank_scores = np.argsort(mean_pagerank_scores)[::-1]
        # plot the 10 most central ROIs
        most_central_ROIs = [labels[sorted_pagerank_scores[i]] for i in range(10)]
        print(most_central_ROIs)

    # saving pagerank scores to file
    if save_name is not None:
        np.save(save_name, pagerank_scores)
    
    return not_converged


def feature_processing(Z_train_ASD, Z_train_TC, Z_test, num_ROIs, labels, num_features, show_best_features=False, independent=False):
    
    # feature selection
    min_pval, p_values = feature_selection(Z_train_ASD, Z_train_TC, num_ROIs, labels, num_features, show_best_features=show_best_features)
    
    if not independent:
        # demean the features
        Z_train_ASD = (Z_train_ASD - np.mean(Z_train_ASD, axis=0)) / np.std(Z_train_ASD, axis=0)
        Z_train_TC = (Z_train_TC - np.mean(Z_train_TC, axis=0)) / np.std(Z_train_TC, axis=0)
        Z_test = (Z_test - np.mean(Z_test, axis=0)) / np.std(Z_test, axis=0)

    # feature extraction
    Z_train_ASD = Z_train_ASD[:, min_pval]
    Z_train_TC = Z_train_TC[:, min_pval]
    Z_test = Z_test[:, min_pval]

    return Z_train_ASD, Z_train_TC, Z_test, p_values

# function called by feature_processing
def feature_selection(Z_train_ASD, Z_train_TC, num_ROIs, labels, num_features, show_best_features=False):

    # calculate the p-value for each ROI
    p_values = []
    for i in range(num_ROIs):
        p_values.append(stats.ttest_ind(Z_train_ASD[:, i], Z_train_TC[:, i])[1])
    p_values = np.array(p_values)

    min_pval = np.argsort(p_values)[:num_features]
    
    if show_best_features:
       for i in min_pval:
          print(labels[i])

    return min_pval, p_values


def debug_tool_plot_2d(Z_train, y_train_sep, Z_test, y_test):

    # Scatter plot for training data
    plt.scatter(Z_train[y_train_sep == 1, 0], Z_train[y_train_sep == 1, 1], c='blue', edgecolors='k', marker='o', label='Train Class 1')
    plt.scatter(Z_train[y_train_sep == 0, 0], Z_train[y_train_sep == 0, 1], c='red', edgecolors='k', marker='o', label='Train Class 0')

    # Scatter plot for testing data
    plt.scatter(Z_test[y_test == 1, 0], Z_test[y_test == 1, 1], c='green', edgecolors='k', marker='x', label='Test Class 1')
    plt.scatter(Z_test[y_test == 0, 0], Z_test[y_test == 0, 1], c='purple', edgecolors='k', marker='x', label='Test Class 0')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
    

def predict_proba(classifer, Z_test, ytest, show_plot=False):
    # Predict probabilities
    y_pred_proba = classifer.predict_proba(Z_test)

    if show_plot:
        # Plot the predicted probabilities

        plt.plot(y_pred_proba[:, 1], 'o', markersize=2, label="Preditions proba")
        plt.plot(ytest, 'o', markersize=2, label="True labels")

        # Show legend
        plt.legend()

        # Display the plot
        plt.show()

    return y_pred_proba

    
def compute_multiplex_eigenvector_centrality(supra_adjacency_matrix, num_ROIs, labels, save_name=None, show_best_features=False):
    num_subjects = int(supra_adjacency_matrix.shape[0]/num_ROIs)
    eigvals, eigvecs = np.linalg.eigh(supra_adjacency_matrix)

    leading_supra_eigenv = eigvecs[:, -1]
    leading_eigen_tens = leading_supra_eigenv.reshape(num_subjects, num_ROIs)
    eigenvector_centrality = np.mean(leading_eigen_tens, axis=0)

    #ordering of the nodes by eigenvector_centrality
    eigenvector_sorted_centrality = np.argsort(eigenvector_centrality)[::-1]
    if show_best_features:
        # plot the 10 most central ROIs
        most_central_ROIs = [labels[eigenvector_sorted_centrality[i]] for i in range(10)]
        print(most_central_ROIs)

    if save_name is not None:
        np.save(save_name, leading_eigen_tens)
    

def compute_independent_eigenvector_centrality(population_corr_matrices, num_ROIs, labels, save_name=None, show_best_features=False):
    from networkx.exception import PowerIterationFailedConvergence

    num_subjects = population_corr_matrices.shape[0]

    eigenvector_centrality = np.zeros((num_subjects, num_ROIs))
    not_converged = []
    for i in tqdm(range(num_subjects), desc="Independent centrality measures", leave=False): 
        try:
            eigenvector_centrality[i, :] = np.array(list(mx.eigenvector_centrality(mx.from_numpy_array(population_corr_matrices[i, :, :]), weight="weight").values()))
        except mx.PowerIterationFailedConvergence:
            eigenvector_centrality[i, :] = None
            not_converged.append(i)

    # remove not converged subjects - may remove if always convergesconverges TODO
    print(f"Subjects for which eigenvector centrality did not converge: {not_converged}, \n Delete them manually from the dataset.")
    #eigenvector_centrality = np.delete(eigenvector_centrality, not_converged, axis=0)

    # mean of each node across subjects
    mean_eigenvector_centrality = np.mean(eigenvector_centrality, axis=0)

    if show_best_features:
        sorted_eigenvector_centrality = np.argsort(mean_eigenvector_centrality)[::-1]
        # plot the 10 most central ROIs
        most_central_ROIs = [labels[sorted_eigenvector_centrality[i]] for i in range(10)]
        print(most_central_ROIs)

    if save_name is not None:
        np.save(save_name, eigenvector_centrality)

    return not_converged


