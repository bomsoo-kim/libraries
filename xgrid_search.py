# Copyright © 2018-2019 Bomsoo Brad Kim, All Rights Reserved.
# Last Update : 04/14/2019
import numpy as np
import pandas as pd

def xgrid_search(params, MIN_MAX = 'min', SCORE_TOLERANCE = 1e-5,  MAX_GRID_LEVEL = 20, MAX_INNER_ROUND = 50, 
                      initialize_cross_grid = False):
    def generate_cgrid(ikeys): # make cross grids
        iparams2 = {k: [0,-1,1] for k in ikeys}
        cgrid = pd.DataFrame(columns = ikeys) # initialize cgrid with empty elements
        for k,i in iparams2.items():
            cgrid = pd.concat([cgrid, pd.DataFrame({k:i})], sort = False) # make cross grids
        cgrid.fillna(0, inplace = True)
        cgrid.drop_duplicates(keep = 'first', inplace = True)
        cgrid.reset_index(drop = True, inplace = True)
        return cgrid

    def trim_grid(grid, ikeys, min_grid, max_grid): # remove duplicate and out-of-range grid points
        i0 = grid[ikeys].duplicated(keep = 'first') # find out which are duplicate elements except the first elements
        i1 = (grid[ikeys] < min_grid[ikeys]).any(axis = 1) # find out rows with less than min indexes
        i2 = (grid[ikeys] > max_grid[ikeys]).any(axis = 1) # find out rows with greater than max indexes
        # grid = pd.concat([grid, i0, i1, i2], sort = False, ignore_index = True, axis = 1)
        grid = grid.loc[~(i0 | i1 | i2)] # remove duplicate and out-of-range grid points
        grid.reset_index(drop = True, inplace = True)
        return grid

    def fill_keys(params2, grid):
        for (k,i) in params2.items(): grid[k] = grid['i_'+k].map({i:value for i, value in enumerate(params2[k])}) # update index
        return grid

    def update_divide_intervals(partype, params2, grid):
        def divide_intervals(a, paramter_type = 'uni'):
            a = np.array(a)
            if paramter_type == 'uni': aa = np.sort(np.unique(np.append(a, (a[1:] + a[:-1])/2)))
            elif paramter_type == 'log': aa = np.sort(np.unique(np.append(a, np.sqrt(a[1:]*a[:-1]))))
            elif paramter_type == 'int': aa = np.sort(np.unique(np.around(np.append(a, (a[1:] + a[:-1])/2)).astype(int)))
            return aa
        for (k,i) in params2.items():
            params2[k] = divide_intervals(i, partype[k]) # divide intervals
            grid['i_'+k] = grid[k].map({value:i for i, value in enumerate(params2[k])}) # update index
        return params2, grid
    
    
    # initial preparation
    partype = {k:i[0] for (k,i) in params.items()} # pick up data type
    params2 = {k:np.sort(np.unique(i[1])) for (k,i) in params.items()} # pick up range
    iparams = {'i_'+k:range(len(i)) for (k,i) in params2.items()} # index for parameters
    pkeys = list(params2.keys())
    ikeys = list(iparams.keys()) # parameter key names
    cgrid = generate_cgrid(ikeys) # cross grid
    
    # initialize grid
    grid = pd.DataFrame(index = pd.MultiIndex.from_product(iparams.values(), names = iparams.keys())).reset_index() # https://stackoverflow.com/questions/13269890/cartesian-product-in-pandas
    grid['func_value'] = np.nan
    grid = fill_keys(params2, grid)
    
    # cross grid setting in the first place
    if initialize_cross_grid:
        params2, grid = update_divide_intervals(partype, params2, grid)
        grid = grid[ikeys].mean(axis = 0).astype(int) + cgrid
        grid[ikeys].astype(int)
        grid['func_value'] = np.nan
        grid = fill_keys(params2, grid)

    # search for the best
    best_fval = pd.DataFrame()
    len_prev_grid = 0
    for nn in range(MAX_GRID_LEVEL):
        for kk in range(MAX_INNER_ROUND):
            # evaluate function
            for i in range(len(grid)):
                if np.isnan(grid.loc[i,'func_value']): # if func_value = nan, then evaluate
                    grid.loc[i,'func_value'] = xgrid_search_target_function(**grid.astype(object).loc[i, pkeys].to_dict())
                    print('> i=%s (level=%s, group=%s); '%(i+1,nn+1,kk+1), grid.astype(object).loc[i, pkeys+['func_value']].to_dict())
            # find best score
            if MIN_MAX == 'max': idx = grid['func_value'].idxmax()
            elif MIN_MAX == 'min': idx = grid['func_value'].idxmin()
            best_grid = grid.loc[idx, ikeys].astype('int') # ensure that the index is integer
            print('  best so far =', grid.astype(object).loc[idx, pkeys+['func_value']].to_dict())
            # explore new grid points
            min_grid = grid.min(axis = 0) # order matters!
            max_grid = grid.max(axis = 0) # order matters!
            grid = pd.concat([grid, best_grid + cgrid], sort = False, ignore_index = True) # attach new rows
            grid = trim_grid(grid, ikeys, min_grid, max_grid) # remove duplicate and out-of-range rows
            grid = fill_keys(params2, grid)
            # stop if there is no NAN value
            if ~grid['func_value'].isna().any(): break

        # stop condition
        if len_prev_grid == len(grid):
            break
        else:
            if MIN_MAX == 'max': 
                idx = grid.loc[len_prev_grid:, 'func_value'].idxmax() # find the best at the latest level
                best_fval = best_fval.append(grid.loc[idx, pkeys+['func_value']]) # save the best row
                best_fval.sort_values(by = 'func_value', ascending  = False, inplace = True)
            elif MIN_MAX == 'min': 
                idx = grid.loc[len_prev_grid:, 'func_value'].idxmin() # find the best at the latest level
                best_fval = best_fval.append(grid.loc[idx, pkeys+['func_value']]) # save the best row
                best_fval.sort_values(by = 'func_value', ascending  = True, inplace = True)
            best_fval.drop_duplicates(keep = 'first', inplace = True)
            if len(best_fval) >= 2:
                best_fval_values = best_fval['func_value'].values
                if abs((best_fval_values[0] - best_fval_values[1])/best_fval_values[0]) < SCORE_TOLERANCE:
                    print('cross grid search completed...'); break;
        len_prev_grid = len(grid)

        # devide intervals
        if nn < MAX_GRID_LEVEL - 1: params2, grid = update_divide_intervals(partype, params2, grid)
                
    if MIN_MAX == 'max': idx = grid['func_value'].idxmax()
    elif MIN_MAX == 'min': idx = grid['func_value'].idxmin()
    best_param = grid.astype(object).loc[idx, pkeys].to_dict()
    
    grid.drop(columns = ikeys, inplace = True)
    return best_param, best_fval, grid

def left_join_crossgridparams_params(cross_grid_params, params):
    for (k,i) in cross_grid_params.items(): 
        cross_grid_params[k][1] = np.sort(np.unique(np.append(cross_grid_params[k][1], params[k])))
    return cross_grid_params

#### Getting Started! ##############################################
def xgrid_search_target_function(**param):
    #--- STRAT user definition ---
    def any_user_function(x,y,z):
        return ((x-2.7)**2) + ((y-3.3)**2) + ((z+2.6)**2) + 1.0
    output = any_user_function(**param)
    #--- END user definition ---
    return output
    
params = { # define the data type and input varaible range
    'x': ['int', [-10, 10]], # uni / log / int
    'y': ['log', [0, 0.01, 10]], # uni / log / int
    'z': ['uni', [-10, -1, 10]]  # uni / log / int
}

param = {'x': 3, 'y': 4, 'z': 5, 'a': 6, 'b': 7} # sample code for how to use left_join_crossgridparams_params
params = left_join_crossgridparams_params(params, param) # sample code for how to use left_join_crossgridparams_params

best_param, best_fval, grid = xgrid_search(params, MIN_MAX = 'min', SCORE_TOLERANCE = 1e-5) # decide on min/max problem and then run!
print('best param = ',best_param)
print(best_fval)
