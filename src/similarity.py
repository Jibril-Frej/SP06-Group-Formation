import numpy as np 


def compute_distance(m, function):
    num_participants = m.shape[0]
    distance_a = np.zeros((num_participants, num_participants))

    for i in range(num_participants):
        v_i = m[i]
        for j in range(i+1, num_participants):
            v_j = m[j]
            same = function(v_i, v_j)
            distance_a[i,j] = same
            distance_a[j,i] = same
    return distance_a      

def function_availability(v_i, v_j):
    return np.sum(v_i & v_j)

def function_strengths(v_i, v_j, difference=3):
    return np.sum(np.abs(v_j - v_i)>difference)

def compute_availability(m_available):
    distance = compute_distance(m_available, function_availability)
    return distance    


def compute_difference(m):
    distance = compute_distance(m, function_strengths)
    return distance  