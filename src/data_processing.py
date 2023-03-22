from .global_var import COLUMNS_TRANSLATED,AVAILABLE, STRENGHTS, INTERESTS, GROUP_SIZE_VAR, INTEREST_VAR
import pandas as pd 
import random
import numpy as np

random_seed = 123
random.seed(random_seed)

def load_data(file_dir):
    df = pd.DataFrame() 
    print('Empry dataframe')
    try:
        df = pd.read_csv(file_dir, sep = ';')
        if len(df.columns) == len(COLUMNS_TRANSLATED):
            df.columns = COLUMNS_TRANSLATED
        else:
            print("Incorrect number of columns in the data file. Please check the file and try again.")
        
    except:
        print("File not found. Please check the file and try again.")

    return df


def simulate_data(num_participants, save = False, 
                  save_dir = '.'):
    
    tandem_preference = [np.random.gamma(1, 2.0)//1%3 for i in range(num_participants)]
    interest_preference = [np.random.gamma(1, 2.0)//1%3 for i in range(num_participants)]

    df_sim = pd.DataFrame({
        GROUP_SIZE_VAR: tandem_preference,
        INTEREST_VAR: interest_preference,
        f'aux_{INTEREST_VAR}': interest_preference
    })

    di = {0: 'Tandem (2 Personen)',
        1: 'Kleingruppe (3-6 Personen)',
        2: 'keine Präferenz'}

    di_interest =  {0: 'Kommunikative Kompetenzen',
        1: 'Führungskompetenzen',
        2:'Digitale Kompetenzen' }

    df_sim = df_sim.replace({GROUP_SIZE_VAR: di,
                             INTEREST_VAR: di_interest})

    for a in AVAILABLE:
        df_sim[a] = [np.random.gamma(1, 1.0)//1%2 for i in range(num_participants)]

    for s in STRENGHTS:
        df_sim[s] = [(np.random.normal(5, 3.0)//1%10)+1 for i in range(num_participants)]

    j = 0
    col = 0
    for interest in INTERESTS :
        if j>3:
            col = 1 
        if j>7:
            col = 2
        df_sim[interest] = [(np.random.normal(7, 2.0)//1%10)+1 if col == interest_preference[i] else 0 for i in range(num_participants) ]
        j+=1
    
    # Shuffle the data
    df_sim = df_sim.sample(frac=1).reset_index(drop=True)
    df_sim['Record ID'] = [i for i in range(num_participants)]
    
    if save:
        df_sim.to_csv(save_dir, index = False, sep=';')

    return df_sim


def split_by_group_size(df):
    df[f'aux_{GROUP_SIZE_VAR}'] = df[GROUP_SIZE_VAR]
    di = {'Tandem (2 Personen)':2,
            'Kleingruppe (3-6 Personen)':6,
            'keine Präferenz':3}

    df.replace({f'aux_{GROUP_SIZE_VAR}': di}, inplace=True)
    df.sort_values(f'aux_{GROUP_SIZE_VAR}', inplace=True)

    df['group_size'] = ['tandem' if i < len(df)//2 else 'kleingruppe' for i in range(len(df))]
    

    # Make odd go to 'kleingruppe'
    df.sort_values(GROUP_SIZE_VAR, inplace=True)
    df_tandem = df[df['group_size']=='tandem']

    for i in ['Kommunikative Kompetenzen', 'Führungskompetenzen', 'Digitale Kompetenzen']:
        df_aux = df_tandem[df_tandem[INTEREST_VAR]==i]
        num = len(df_aux)
        if num%2 != 0: # uneven 
            last_index = df_aux.index[-1]
            df.loc[last_index, 'group_size'] = 'kleingruppe'


    # Report the number of participants in each group
    print(df[[GROUP_SIZE_VAR,'group_size']].value_counts())


    df_tandem = df[df['group_size'] == 'tandem'].copy()
    df_kleingruppe = df[df['group_size'] == 'kleingruppe'].copy()

    return df_tandem, df_kleingruppe 
