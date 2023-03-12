import json
import random
import numpy as np

import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.cluster import SpectralClustering

random.seed(123)


def load_data(data_path, data_file, label_file, translation_file, verbose = 0):
    df = pd.read_csv(data_file, sep = ';')
    df_labels = pd.read_csv(label_file, sep = ';')
    df_translated = df_labels
    
    with open(translation_file) as f:
        df_translated.columns = json.load(f)
    
    df_translated =df_translated[~df_translated['Conduct Difficult Conversations'].isna()] 
    
    # Separate in two groups
    df_translated.insert(len(df_translated.columns), 'treatment', [random.randint(0, 1) for i in range(len(df_translated))])
    return df_translated

    
    
# Constrain: difference among persons shouldn''t be too high (max 3)
def get_labels(m_interests, metric= 'euclidean',gamma = 0.01):
    distance = squareform(pdist(m_interests, metric=metric))
    similarity = np.exp(-gamma * distance ** 2)
    labels = SpectralClustering(n_clusters=2,random_state=123,affinity='precomputed').fit_predict(similarity)
    return labels

def hierarchical_spectral(m_matrix , labels, metric= 'euclidean',gamma = 0.01, offset = 2, verbose = 0):    
    
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    if verbose:
        print(unique_labels, counts_labels)
    labels_to_explore = unique_labels

    while len(labels_to_explore)>= 1:
        if verbose:
            print('Labels to explore: ', labels_to_explore)
        i = labels_to_explore[0]
        mask = labels==i
        size = np.sum(mask)
        if size >= 4:
            # see if we can split
            m_aux = m_matrix[mask]
            new_labels = get_labels(m_aux, metric, gamma) + offset

            new_unique_labels, new_counts_labels = np.unique(new_labels, return_counts=True)
            if np.min(new_counts_labels)>=2:
                # We acept the clustering, we upload the labels 
                labels[mask] = new_labels 
                
                offset += 2 # We move the offset
                # We add the new labels as labels to explore to the list
                labels_to_explore = np.union1d(labels_to_explore, np.unique(new_labels))
                labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
                if verbose:
                    print(f'Adding labels and removes current {i}. Updated list: {labels_to_explore}')
                unique_labels, counts_labels = np.unique(labels, return_counts=True)
                if verbose:
                    print(f"Unique labels {unique_labels} with counts {counts_labels}")
            else:
                # We do not accept, keep the group as it is. Not divisible. 
                labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
                if verbose:
                    print(f"We do not accept the new labels. Counts {new_counts_labels}")
                    print('Removing labels: ', labels_to_explore)
        else:
            labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
            if verbose:
                print(f"Small group for {i}. Size = {size}")
                print('Removing labels: ', labels_to_explore)


    return labels

def get_similarity_clusters(m_matrix, metric= 'euclidean',gamma = 0.01):
    distance = squareform(pdist(m_matrix, metric=metric))
    similarity = np.exp(-gamma * distance ** 2)
    labels = SpectralClustering(n_clusters=2,random_state=123,affinity='precomputed').fit_predict(similarity)
    
    new_labels = hierarchical_spectral(m_matrix , labels, metric= 'euclidean',gamma = 0.01)
    return new_labels




def get_distance_matrix(m_aux_strenghts, m_aux_interest, verbose = 0):
    num_participants = m_aux_strenghts.shape[0]
    if verbose:
        print(f" participants {num_participants}")
    # create matrix
    distance_results = np.zeros([num_participants,num_participants], dtype=np.float64)
    for i in range(num_participants):
        # get interest and strength
        interest_i = m_aux_interest[i]
        strength_i = m_aux_strenghts[i]
        for j in range(i+1, num_participants):
                strength_j = m_aux_strenghts[j]
                interest_j = m_aux_interest[j]
                dist_a = euclidean(interest_i, strength_j)
                dist_b = euclidean(interest_j, strength_i)
                dist = np.mean([dist_a, dist_b])
                distance_results[i,j] = dist
                distance_results[j,i] = dist
    return distance_results



def get_contrasting_clusters(m_comp_strenghts, m_comp_interest, metric= 'euclidean',gamma = 0.01, verbose = 0):
    distance_matrix = get_distance_matrix(m_comp_strenghts, m_comp_interest, verbose = verbose)
    similarity = np.exp(-gamma * distance_matrix ** 2)
    labels = SpectralClustering(n_clusters=2,random_state=123,affinity='precomputed').fit_predict(similarity)
    
    new_labels = hierarchical_spectral_comp(m_comp_strenghts, m_comp_interest , labels, metric= 'euclidean',gamma = 0.01, verbose = verbose)
    return new_labels


def get_labels_comp(m_aux_strenghts, m_aux_interest, metric= 'euclidean',gamma = 0.01):
    distance_matrix_comp = get_distance_matrix(m_aux_strenghts, m_aux_interest)
    similarity = np.exp(-gamma * distance_matrix_comp ** 2)
    labels = SpectralClustering(n_clusters=2,random_state=123,affinity='precomputed').fit_predict(similarity)
    return labels

def hierarchical_spectral_comp(m_comp_strenghts, m_comp_interest , labels, metric= 'euclidean',gamma = 0.01, offset_num = 2, verbose = 0):    
    offset = offset_num
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    if verbose:
        print(unique_labels, counts_labels)
    labels_to_explore = unique_labels

    while len(labels_to_explore)>= 1:
        if verbose:
            print('Labels to explore: ', labels_to_explore)
        i = labels_to_explore[0]
        mask = labels==i
        if verbose:
                print(f"labels {labels}")
        size = np.sum(mask)
        if verbose:
            print(f'Exploring label {i}, with size {size}')
        if size >= 4:
            # see if we can split
            m_aux_strenghts = m_comp_strenghts[mask]
            m_aux_interest = m_comp_interest[mask]
            
            new_labels = get_labels_comp(m_aux_strenghts, m_aux_interest, metric, gamma) + offset
            new_unique_labels, new_counts_labels = np.unique(new_labels, return_counts=True)
            
            if np.min(new_counts_labels)>=2:
                # We acept the clustering, we upload the labels 
                labels[mask] = new_labels 
                offset += 2 # We move the offset
                # We add the new labels as labels to explore to the list
                labels_to_explore = np.union1d(labels_to_explore, np.unique(new_labels))
                labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
                if verbose:
                    print(f'Adding labels and removes current {i}. Updated list: {labels_to_explore}')
                unique_labels, counts_labels = np.unique(labels, return_counts=True)
                if verbose:
                    print(f"Unique labels {unique_labels} with counts {counts_labels}")
            else:
                # We do not accept, keep the group as it is. Not divisible. 
                labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
                if verbose:
                    print(f"We do not accept the new labels. Counts {new_counts_labels}")
                    print('Removing labels: ', labels_to_explore)
        else:
            labels_to_explore = np.setdiff1d(labels_to_explore, np.array([i]))
            if verbose:
                print(f"Small group for {i}. Size = {size}")
                print('Removing labels: ', labels_to_explore)


    return labels





def group_by_similarity(df_translated, similar_interest_file, availabilities_file, verbose = 0):
    df_similar = df_translated[df_translated['treatment']==1]
    
    with open(similar_interest_file) as f:
        similar_interest_list = json.load(f)

    df_similar_interests = df_similar[similar_interest_list].fillna(-1)
    m_interests = np.matrix(df_similar_interests)
    m_interests[m_interests<=3] = -1

    labels = get_similarity_clusters(m_interests , metric= 'euclidean', gamma = 0.01)    
    df_similar.insert(len(df_similar.columns), 'labels_similarity', labels)

    with open(availabilities_file) as f:
        availabilities_list = json.load(f)

    df_similar_available = df_similar[availabilities_list]
    df_similar_available = df_similar_available.replace('Unchecked',0).replace('Checked',1)
    m_available = np.array(df_similar_available)

    available_labels = hierarchical_spectral(m_available , labels, metric= 'hamming',gamma = 0.01, offset=10, verbose = verbose)
    df_similar.insert(len(df_similar.columns), 'available_labels', available_labels)

    df_similar.insert(len(df_similar.columns), 'study_group', df_similar['available_labels'] + 1000)

    display(df_similar[['treatment', 'available_labels', 'study_group', 'labels_similarity','Record ID']])

    return df_similar



def group_by_complementarity(df_translated, comp_interest_file, comp_strength_file, availabilities_file, verbose = 0):
    df_comp = df_translated[df_translated['treatment']==0]
    
    with open(comp_interest_file) as f:
        comp_interest_list = json.load(f)

    m_comp_interest =  np.array(df_comp[comp_interest_list].fillna(-1))

    with open(comp_strength_file) as f:
        comp_strength_list = json.load(f)


    m_comp_strength =  np.array(df_comp[comp_strength_list].fillna(-1))

    labels = get_contrasting_clusters(m_comp_strength, m_comp_interest, metric= 'euclidean', gamma = 0.01)
    df_comp.insert(len(df_comp.columns), 'labels_complementary', labels)

    with open(availabilities_file) as f:
        availabilities_list = json.load(f)

    df_comp_available = df_comp[availabilities_list]
    df_comp_available = df_comp_available.replace('Unchecked', 0).replace('Checked',1)
    m_available_comp = np.array(df_comp_available)

    available_labels = hierarchical_spectral(m_available_comp , labels, metric= 'hamming', gamma = 0.01, offset=10, verbose = verbose)
    df_comp.insert(len(df_comp.columns), 'available_labels', available_labels)

    df_comp.insert(len(df_comp.columns), 'study_group', df_comp['available_labels'] + 2000)


    display(df_comp[['treatment', 'available_labels', 'study_group', 'labels_complementary','Record ID']])
    return df_comp




def get_groups(df_similar, df_comp):
    df_groups = pd.concat([df_comp[['treatment', 'study_group','Record ID']], df_similar[['treatment', 'study_group','Record ID']]], axis=0)
    df_groups['treatment'] = df_groups['treatment'].replace(0, 'complementary').replace(1, 'similar')
    return df_groups