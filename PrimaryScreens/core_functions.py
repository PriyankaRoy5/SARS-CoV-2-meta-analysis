import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anchors
from poola import core as pool
from sklearn.metrics import auc

## Reformatting functions 
##

def clean_Sanjana_data(df, guide_col='Guide', library = False):
    '''
    Input: 1. df: Reads dataframe with guide_col and data columns 
           2. guide_col: Formatted as 'library_guide_gene' (e.g. 'HGLibA_00001_A1BG')
            

    Output: df_clean: Dataframe with columns 'Guide', 'Gene Symbol', 'Reads'
    '''
    df_clean = df.rename(columns={guide_col:'old_Guide'})
    library_list = []
    guide_list = []
    gene_list = []
    for i, row in enumerate(df_clean.loc[:,'old_Guide']):
        split_row = row.split('_')
        library = split_row[0]
        library_list.append(library)
        guide = split_row[1]
        guide_list.append(guide)
        gene = split_row[2]
        gene_list.append(gene)

    df_clean['Library'] = pd.Series(library_list)
    df_clean['Guide#'] = pd.Series(guide_list)
    df_clean['Guide'] = df_clean[['Library','Guide#']].apply(lambda x: '_'.join(x.dropna().values), axis=1)
    df_clean['Gene Symbol'] = pd.Series(gene_list)

    df_clean = df_clean.drop(['Library', 'Guide#','old_Guide'], axis = 1)
    
    # Reorder columns so Guide, Gene Symbol then data columns
    data_cols = [col for col in df.columns if col != guide_col]
    col_order = ['Guide','Gene Symbol'] + data_cols
    df_clean = df_clean[col_order]
    
    return df_clean

def merge_dict_dfs(dictionary, merge_col = 'Gene Symbol', merge_how = 'outer', suffixes = ['_x', '_y']):
    '''
    Input: 1. dictionary: dictionary containing dataframes 
           2. merge_col: name of column on which dataframes will be merged (default = 'Gene Symbol')
           3. merge_how: type of merge (default = 'outer')
           4. suffixes: suffixes if two columns have the same name in dataframes being merged (default = ['_x','_y'])
            
    Output: merge1: merged dataframe 
    '''
    merge1 = pd.DataFrame()
    keys = []
    for df_name in dictionary.keys():
        keys.append(df_name)
    for i, df_name in enumerate(keys):
        current_df = dictionary[df_name]
        if (i+1 < (len(keys))): #stop before last df
            next_df_key = keys[i+1]
            next_df = dictionary[next_df_key]
            # merge dfs 
            if merge1.empty:  # if merged df does not already exist 
                merge1 = pd.merge(current_df, next_df, on = merge_col, how = merge_how, suffixes = suffixes)
                #print(merge1.columns)
            else: #otherwise merge next_df with previous merged df
                new_merge = pd.merge(merge1, next_df, on = merge_col, how = merge_how)
                merge1 = new_merge
    return merge1

def convertdftofloat(df):
    '''
    Converts df data column type into float 
    Input:
    1. df: data frame
    '''
    for col in df.columns[1:]:
        df[col] = df[col].astype(float) #convert dtype to float 
    return df

## QC functions
## 

def get_lognorm(df, cols = ['Reads'], new_col = ''):
    '''
    Inputs: 
    1. df: clean reads dataframe
    2. cols: list of names of column containing data used to calculate lognorm (default = ['Reads'])
    3. new_col: lognorm column name (optional) 
    Output: New dataframe with columns 'Gene Symbol', '[col]_Lognorm' (default = 'Reads_lognorm')
    '''
    df_lognorm = df.copy().drop(cols, axis = 1)
    for c in cols:
        df_lognorm[c+'_lognorm'] = pool.lognorm(df[c])
    return df_lognorm

def calculate_lfc(lognorm_df, target_cols, ref_col = 'pDNA_lognorm'): 
    '''
    Inputs:
    1. lognorm_df: Dataframe containing reference and target lognorm columns 
    2. target_cols: List containing target column name(s) (lognorm column(s) for which log-fold change should be calculated)
    3. ref_col: Reference column name (lognorm column relative to which log-fold change should be calculated)(default ref_col = 'pDNA_lognorm')
    Outputs:
    1. lfc_df: Dataframe containing log-fold changes of target columns 
    '''
    #input df with lognorms + pDNA_lognorm
    lfc_df = pool.calculate_lfcs(lognorm_df=lognorm_df,ref_col=ref_col, target_cols=target_cols)
    for col in target_cols: #rename log-fold change column so doesn't say "lognorm"
        lfc_col_name = col.replace('lognorm', 'lfc') 
        lfc_df = lfc_df.rename(columns = {col:lfc_col_name})
    return lfc_df

def get_controls(df, control_name = ['NonTargeting'], separate = True):
    '''
    Inputs:
    1. df: Dataframe with columns "Gene Symbol" and data 
    2. control_name: list containing substrings that identify controls 
    3. separate: determines whether to return non-targeting and intergenic controls separately (default = True)
    Outputs:
    1. control: Dataframe containing rows with Gene Symbols including control string specified in control_name 
    OR 2. control_dict: If separate and multiple control names, dictionary containing dataframes 
    OR 3. all_controls: If separate = False and multiple control names, concatenated dataframes in control_dict 
    '''
    if len(control_name) == 1:
        control = df[df['Gene Symbol'].str.contains(control_name[0], na=False)]
        return control
    else:
        control_dict = {}
        for i, ctrl in enumerate(control_name):
            control_dict[ctrl] = df[df['Gene Symbol'].str.contains(ctrl, na=False)]
        if separate: 
            return control_dict
        else:
            all_controls = pd.concat(list(control_dict.values()))
            return all_controls
        
def get_gene_sets():
    '''
    Outputs: essential and non-essential genes as defined by Hart et al. 
    '''
    ess_genes = pd.read_csv('https://raw.githubusercontent.com/gpp-rnd/genesets/master/human/essential-genes-Hart2015.txt', sep='\t', header=None)
    ess_genes.columns = ['Gene Symbol']
    ess_genes['ess-val'] = [1]*len(ess_genes)
    non_ess = pd.read_csv('https://raw.githubusercontent.com/gpp-rnd/genesets/master/human/non-essential-genes-Hart2014.txt', sep='\t', header=None)
    non_ess.columns = ['Gene Symbol']
    non_ess['non-ess-val'] = [1]*len(non_ess)
    return ess_genes, non_ess

def merge_gene_sets(df):
    '''
    Input:
    1. df: data frame from which ROC-AUC is being calculated 
    Output:
    1. df: data frame with binary indicators for essential and non-essential genes 
    '''
    ess_genes, non_ess = get_gene_sets()
    df = pd.merge(df, ess_genes, on='Gene Symbol', how='left')
    df['ess-val'] = df['ess-val'].fillna(0)
    df = pd.merge(df, non_ess, on='Gene Symbol', how='left')
    df['non-ess-val'] = df['non-ess-val'].fillna(0)
    return df

def get_roc_auc(df, col):
    '''
    Inputs:
    1. df: data frame from which ROC-AUC is being calculated 
    2. col: column with data for which ROC-AUC is being calculated
    Outputs: 
    1. roc_auc: AUC value where true positives are essential genes and false positives are non-essential
    2. roc_df: dataframe used to plot ROC-AUC curve 
    '''
    df = df.sort_values(by=col)
    df['ess_cumsum'] = np.cumsum(df['ess-val'])
    df['non_ess_cumsum'] = np.cumsum(df['non-ess-val'])
    df['fpr'] = df['non_ess_cumsum']/(df['non_ess_cumsum'].iloc[-1])
    df['tpr'] = df['ess_cumsum']/(df['ess_cumsum'].iloc[-1])
    df.head()
    roc_auc = auc(df['fpr'],df['tpr'])
    roc_df = pd.DataFrame({'False_Positive_Rate':list(df.fpr), 'True_Positive_Rate':list(df.tpr)})
    return roc_auc, roc_df

## Plotting functions
##
 
def pair_cols(df, initial_id, res_id, sep = '_', col_type = 'lfc'): #if more than one set of initial/resistant pop pairs, sharex = True, store pairs in list
    '''
    Inputs: 
    1. df: Dataframe containing log-fold change values and gene symbols 
    2. initial_id: string identifying initial column names (default: 'control'), only used if multiple subplots
    3. res_id: string identifying resistant column names (default: 'MOI'), only used if multiple subplots
    4. sep: character separator in column name 
    3. col_type: string in names of columns containing data to be plotted (default: 'lfc')
    Outputs: 
    1. sharex: if number of pairs greater than 1 indicating multiple subplots
    2. pairs: pairs of initial and resistant populations as list of lists 
    '''
    cols = [col for col in df.columns if col_type in col]
    pairs = [] #list of lists: ini/res pop pairs
    sharex = False
    if len(cols) > 2: #if more than one set of initial/resistant pop pairs 
        
        for index, col in enumerate(cols):
            pair = []
            if initial_id in col: #find corresponding resistant pop
                pair.append(col)
                res_pop = [col for col in cols if res_id in col]

                for col in res_pop:
                    pair.append(col)
                    
                pairs.append(pair) #add to list of pairs (list of lists)
        if len(pairs) > 1:
            sharex = True # set sharex parameter for subplot 
        return sharex, pairs

    else: #if only one pair of initial/resistant pops
        sharex = False
        pairs.append(cols)
        return sharex, pairs

def lfc_dist_plot(chip_lfc, initial_id=None, res_id=None, paired_cols=None, col_sep = '_', filename = '', figsize = (6,4)): #kde plots of population distribution (initial, resistant)
    '''
    Inputs: 
        1. chip_lfc: Dataframe containing log-fold change values and gene symbols 
        Option 1:
        2. initial_id: substring in names of column containing log-fold changes of uninfected population
        3. res_id: substring in names of column containing log-fold changes of infected population
        Option 2:
        4. paired_cols: if using modified pair_cols function but same two outputs of sharex, lfc_pairs
        5. filename: string for file name when saving figure 
        6. figsize: default (6,4)
                            
    Outputs: kde plots of population distribution (initial, resistant)
    '''
    if not paired_cols:    
        sharex, lfc_pairs = pair_cols(chip_lfc, initial_id = initial_id, res_id = res_id, sep = col_sep)
    else:
        sharex, lfc_pairs = paired_cols
        
    fig, ax = plt.subplots(nrows = len(lfc_pairs), ncols = 1, sharex = sharex, figsize = figsize)
    
    i = 0 # ax index if have to plot multiple axes
    for k,c in enumerate(lfc_pairs):
        
        for l, c1 in enumerate(c):
            #title ex. Calu-3 Calabrese A screen 1, (k+1 = screen #)
            if not filename:
                title = ' '.join(c1.split(' ')[:3]) + ' (populations)'
            else:
                title = filename
            if l==0:
                label1 = c1
            else:
                label1 = c1
                
            if sharex: #if multiple axes, ax = ax[i]
                chip_lfc[c1].plot(kind='kde',c=sns.color_palette('Set2')[l],label=label1, ax=ax[i], legend=True)
                t = ax[i].set_xlabel('Log-fold changes') 
                t = ax[i].set_title(title)
                ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else: 
                chip_lfc[c1].plot(kind='kde',c=sns.color_palette('Set2')[l],label=label1, ax=ax, legend=True)
                t = ax.set_xlabel('Log-fold changes')
                t = ax.set_title(title)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i+=1
        sns.despine()   
    
    
#Control distributions
def control_dist_plot(chip_lfc, control_name, filename, gene_col = 'Gene Symbol', initial_id=None, res_id=None, paired_cols=None, col_sep = '_', figsize = (6,4)): 
    '''
    Inputs: 
    1. chip_lfc: annotated lfc data frame
    2. control_name: list containing strings identifying controls 
    3. initial_id: string identifying initial column names
    4. res_id: string identifying resistant column names
    5. filename: filename for saving figure
    6. figsize: default (6, 4)
    Outputs: kde plots of control distributions (initial, resistant)
    
    '''
    if not paired_cols:    
        sharex, lfc_pairs = pair_cols(chip_lfc, initial_id = initial_id, res_id = res_id, sep = col_sep)
    else:
        sharex, lfc_pairs = paired_cols
    controls = get_controls(chip_lfc, control_name)
    nrows = len(lfc_pairs)
    
    fig, ax = plt.subplots(nrows = nrows, ncols = 1, sharex = sharex, figsize = figsize)
    i = 0 # ax index if have to plot multiple axes
    for k,c in enumerate(lfc_pairs): # k=screen, c=ini, res pair 
        for l, c1 in enumerate(c): # l = ini or res, c1 = pop label 
            title = c1 + ' (controls)'
            pop_label = c1.split(' ')[0] #labels 'initial' or 'resistant'
            #Plot same screen on same subplot 
            if sharex: #if multiple axes, ax = ax[i]
                controls[c1].plot(kind='kde',c=sns.color_palette('Set2')[l],label=control_name[0] +' ('+pop_label+')', ax=ax[i], legend=True)
                ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                sns.despine()
                t = ax[i].set_xlabel('Log-fold changes')
                t = ax[i].set_title(title)
            else: 
                controls[c1].plot(kind='kde',c=sns.color_palette('Set2')[l],label=control_name[0]+ ' ('+pop_label+')', ax=ax, legend=True)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                sns.despine()
                t = ax.set_xlabel('Log-fold changes')
                t = ax.set_title(title)
        i+=1 #switch to next subplot for next screen

        sns.despine()
    
        
## Residual functions 

def run_guide_residuals(lfc_df, initial_id=None, res_id=None, paired_cols = None):
    '''
    Calls get_guide_residuals function from anchors package to calculate guide-level residual z-scores
    Input:
    1. lfc_df: data frame with log-fold changes (relative to pDNA)
    
    '''
    lfc_df = lfc_df.drop_duplicates()
    if not paired_cols:
        paired_lfc_cols = pair_cols(lfc_df, initial_id, res_id)[1] #get lfc pairs 
    else:
        paired_lfc_cols = paired_cols
    #reference_df: column1 = modifier condition, column2 = unperturbed column
    ref_df = pd.DataFrame(columns=['modified', 'unperturbed'])
    row = 0 #row index for reference df 
    for pair in paired_lfc_cols:
        #number of resistant pops in pair = len(pair)-1
        res_idx = 1 
        #if multiple resistant populations, iterate 
        while res_idx < len(pair): 
            ref_df.loc[row, 'modified'] = pair[res_idx]
            ref_df.loc[row, 'unperturbed'] = pair[0]
            res_idx +=1 
            row +=1
    print(ref_df)
    #input lfc_df, reference_df 
    #guide-level
    residuals_lfcs, all_model_info, model_fit_plots = anchors.get_guide_residuals(lfc_df, ref_df)
    return residuals_lfcs, all_model_info, model_fit_plots


def format_gene_residuals(df, guide_min, guide_max, ascending = False, suffixes = ['_x', '_y']):
    '''
    Inputs: 
    1. df: gene_residuals output df 
    2. guide_min: min number of guides per gene to filter df
    3. guide_max: max number of guides per gene to filter df
    4. ascending: direction to sort df 
    Outputs:
    1. df_z: dataframe with the following columns: 
            -Gene Symbol
            -residual_zscore: residual_zscores averaged across conditions 
            -Rank_residual_zscore: 
    '''
    df = df[(df['guides']>=guide_min) & (df['guides']<=guide_max)]
    if 'condition' in df.columns: 
        conditions = list(set(df.loc[:, 'condition']))
        print(conditions)
        if len(conditions) > 1:
            df_z = df[['condition', 'Gene Symbol', 'residual_zscore']]
            condition_dict = {}
            for i, c in enumerate(conditions):
                print(c)
                condition_dict[c] = df_z[df_z['condition'] == c]

            merged_df_z = merge_dict_dfs(condition_dict, suffixes=suffixes)
            merged_df_z['residual_zscore_avg'] = merged_df_z.mean(axis = 1)
            df_z = merged_df_z.copy()[['Gene Symbol', 'residual_zscore_avg']]
            df_z_ranked = df_z.copy()
            df_z_ranked.loc[:,'Rank_residual_zscore_avg'] = df_z.loc[:,'residual_zscore_avg'].copy().rank(method='min', ascending=ascending)
            df_z_ranked = df_z_ranked.sort_values(by= 'Rank_residual_zscore_avg')

        else:
            df_z = df[['Gene Symbol', 'residual_zscore']]
            df_z_ranked = df_z.copy()
            df_z_ranked.loc[:,'Rank_residual_zscore'] = df_z.loc[:,'residual_zscore'].copy().rank(method='min', ascending=ascending)
            df_z_ranked = df_z_ranked.sort_values(by= 'Rank_residual_zscore')
    else:
        df_z = df[['Gene Symbol', 'residual_zscore']]
        df_z_ranked = df_z.copy()
        df_z_ranked.loc[:,'Rank_residual_zscore'] = df_z.loc[:,'residual_zscore'].copy().rank(method='min', ascending=ascending)
        df_z_ranked = df_z_ranked.sort_values(by= 'Rank_residual_zscore')

    return df_z_ranked




