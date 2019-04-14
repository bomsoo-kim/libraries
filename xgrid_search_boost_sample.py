# Copyright © 2018-2019 Bomsoo Brad Kim, All Rights Reserved.
# Last Update : 04/14/2019
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn import metrics   #Additional scklearn functions

def xgrid_search_xgboost(train, test, features, target, params_schedule, mlmodel, eval_metric, MIN_MAX): 
    #--- xgrid_search algorithm --------------------------------------------
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
    
    #--- evaluation function with xgboost, ligthgbm ------------------------------
    def xgrid_search_target_function(**param): # define user function with all the input variables
        #--- START: user defined function -----------------------------
        mlmodel.set_params(**param) # update some parameters
        test_pred, valid_score = user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX)
        #--- END ------------------------------------------------------
        return valid_score

    #-----------------------------------------------------------------------
    for params in params_schedule:
    #     params = left_join_crossgridparams_params(params, mlmodel.get_xgb_params()) # ensure that the latest xgmodel values are included
        best_param, best_fval, grid = xgrid_search(params, MIN_MAX = MIN_MAX, SCORE_TOLERANCE = 1e-5) # decide on min/max problem and then run!
        mlmodel.set_params(**best_param) # update some parameters with the best so far
        print('best param = ',best_param)
        print(best_fval)    
        print(mlmodel)
    
    test_pred, valid_score = user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = True)
    print('final validatoin score = ',valid_score)
    print(mlmodel) # final model confirmation
    
    return mlmodel, test_pred

#--- user_defined_eval_function --------------------------------------------------------------
# validatoin approach: XGBoost
# def user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = False):
#     X_train, X_valid, y_train, y_valid = train_test_split(train[features], train[target], test_size=0.20, random_state=42)
#     mlmodel.set_params(n_estimators = 10000) # initialize n_estimators
#     mlmodel.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], eval_metric = eval_metric, 
#                 early_stopping_rounds = 50, verbose = False) #Fit the algorithm on the data
#     if MIN_MAX == 'max': idx = np.array(list(mlmodel.evals_result()['validation_1'].values())).argmax()
#     elif MIN_MAX == 'min': idx = np.array(list(mlmodel.evals_result()['validation_1'].values())).argmin()
#     mlmodel.set_params(n_estimators = idx + 1) # update n_estimators
#     train_score = np.array(list(mlmodel.evals_result()['validation_0'].values())).squeeze()[idx]
#     valid_score = np.array(list(mlmodel.evals_result()['validation_1'].values())).squeeze()[idx]
#     test_pred = mlmodel.predict(test[features])     # computationally not expensive
#     return test_pred, valid_score
    
# cross validation 1: XGBoost
# def user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = False):
#     dtrain = xgb.DMatrix(train[features], label = train[target], missing = np.nan) # missing value handling: https://www.youtube.com/watch?v=cVqDguNWh4M
#     cvoutp = xgb.cv(mlmodel.get_xgb_params(), dtrain, num_boost_round = 10000, verbose_eval =  False, 
#                       nfold = 5, metrics = eval_metric, early_stopping_rounds = 50) # early_stopping_rounds
#     mlmodel.set_params(n_estimators = cvoutp.shape[0]) # update n_estimator 
#     train_score = cvoutp.tail(1)[cvoutp.columns[cvoutp.columns.str.contains('train-.+-mean', regex=True)]].squeeze()
#     valid_score = cvoutp.tail(1)[cvoutp.columns[cvoutp.columns.str.contains('test-.+-mean', regex=True)]].squeeze()

#     if predict_test_output == True:
#         mlmodel.fit(train[features], train[target].values.ravel(), eval_metric = eval_metric) #Fit the algorithm on the data
#         test_pred = mlmodel.predict(test[features])    
#     else: 
#         test_pred = []
#     return test_pred, valid_score    
    
# cross validation 2: XGBoost, LightGBM
def user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = False):
#         folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=2319) # cv n-fold
    folds = KFold(n_splits=5, shuffle=False, random_state=2319) # cv n-fold
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for n, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, train[target].values)):
        X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx][target].values.ravel()
        X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx][target].values.ravel()

        mlmodel.set_params(n_estimators = 10000) # initialize n_estimators
        mlmodel.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], eval_metric = eval_metric, 
                    early_stopping_rounds = 50, verbose = False) #Fit the algorithm on the data

        if eval_metric == 'auc': 
            oof[val_idx] = mlmodel.predict_proba(X_valid)[:,1]
            test_pred += mlmodel.predict_proba(test[features])[:,1] / folds.n_splits
        if eval_metric == 'rmse': 
            oof[val_idx] = mlmodel.predict(X_valid)
            test_pred += mlmodel.predict(test[features]) / folds.n_splits

    if eval_metric == 'auc': 
        valid_score = metrics.roc_auc_score(train[target], oof)
    if eval_metric == 'rmse': 
        valid_score = np.sqrt(metrics.mean_squared_error(train[target], oof))

    return test_pred, valid_score

#--- LigthGBM -----------------------------------------------------------------
# eval_metric = 'auc'; MIN_MAX = 'max'; mlmodel = LGBMClassifier(learning_rate = 0.1, n_jobs = 4, seed = 123) # classification 
# eval_metric = 'rmse'; MIN_MAX = 'min'; mlmodel = LGBMRegressor(learning_rate = 0.1, n_jobs = 4, seed = 123) # regression
    
# params_schedule = [
# #     {'learning_rate': ['log', [0.1]]},
    
# #     {'min_child_weight': ['log', [1, 10, 100, 1000]]}, # uni / log / int
# #     {'gamma': ['log', [0, 0.001, 1, 1000]]}, # uni / log / int
# #     {'subsample': ['uni', [0.2, 0.4, 0.6, 0.8, 1.0]],  # uni / log / int
# #      'colsample_bytree': ['uni', [0.2, 0.4, 0.6, 0.8, 1.0]]},  # uni / log / int
# #     {'reg_alpha': ['log', [0, 0.001, 1, 1000]],
# #      'reg_lambda': ['log', [1, 10, 100, 1000]]},  # uni / log / int
                       
# #     {'learning_rate': ['log', [0.005, 0.1]]} # uni / log / int
# ]

#--- XGBoost -----------------------------------------------------------------
# eval_metric = 'auc'; MIN_MAX = 'max'; mlmodel = XGBClassifier(learning_rate = 0.1, n_jobs = 4, seed = 123) # classification 
eval_metric = 'rmse'; MIN_MAX = 'min'; mlmodel = XGBRegressor(learning_rate = 0.1, n_jobs = 4, seed = 123) # regression

params_schedule = [
#     {'learning_rate': ['log', [0.1]]},
    
#     {'max_depth': ['int', [3, 5, 7, 9, 11]], # uni / log / int
#      'min_child_weight': ['log', [1, 10, 100, 1000]]}, # uni / log / int
#     {'gamma': ['log', [0, 0.001, 1, 1000]]}, # uni / log / int
#     {'subsample': ['uni', [0.2, 0.4, 0.6, 0.8, 1.0]],  # uni / log / int
#      'colsample_bytree': ['uni', [0.2, 0.4, 0.6, 0.8, 1.0]]},  # uni / log / int
#     {'reg_alpha': ['log', [0, 0.001, 1, 1000]],
#      'reg_lambda': ['log', [1, 10, 100, 1000]]},  # uni / log / int
                       
#     {'learning_rate': ['log', [0.005, 0.1]]} # uni / log / int
]

mlmodel, test_pred = xgrid_search_xgboost(train, test, features, target, params_schedule, mlmodel, eval_metric, MIN_MAX)
