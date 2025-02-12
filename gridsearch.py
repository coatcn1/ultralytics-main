from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.linear_model import Lasso as cuLasso
from cuml.ensemble import ExtraTreesRegressor as cuET
from cuml.ensemble import GradientBoostingRegressor as cuGB
from cuml.ensemble import AdaBoostRegressor as cuAB
from cuml.model_selection import GridSearchCV

# RandomForestRegressor
rf_model = cuRF()
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
print("Best parameters for RandomForestRegressor: ", rf_grid_search.best_params_)

# Lasso
lasso_model = cuLasso()
lasso_grid_search = GridSearchCV(estimator=lasso_model, param_grid=param_grid_lasso, cv=5, n_jobs=-1)
lasso_grid_search.fit(X_train, y_train)
print("Best parameters for Lasso: ", lasso_grid_search.best_params_)

# ExtraTreesRegressor
et_model = cuET()
et_grid_search = GridSearchCV(estimator=et_model, param_grid=param_grid_et, cv=5, n_jobs=-1)
et_grid_search.fit(X_train, y_train)
print("Best parameters for ExtraTreesRegressor: ", et_grid_search.best_params_)

# GradientBoostingRegressor
gb_model = cuGB()
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, n_jobs=-1)
gb_grid_search.fit(X_train, y_train)
print("Best parameters for GradientBoostingRegressor: ", gb_grid_search.best_params_)

# XGBRegressor
xgb_model = XGBRegressor(tree_method='gpu_hist')  # 使用GPU加速
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
print("Best parameters for XGBRegressor: ", xgb_grid_search.best_params_)

# AdaBoostRegressor
ab_model = cuAB()
ab_grid_search = GridSearchCV(estimator=ab_model, param_grid=param_grid_ab, cv=5, n_jobs=-1)
ab_grid_search.fit(X_train, y_train)
print("Best parameters for AdaBoostRegressor: ", ab_grid_search.best_params_)