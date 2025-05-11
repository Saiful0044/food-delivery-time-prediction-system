import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import yaml

target_col = 'time_taken'
# create logger
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# load data
def load_data(data_path: Path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"File not found at {data_path}")
        raise
    return df
# split x and y
def split_X_and_y(data: pd.DataFrame, target: str)-> pd.DataFrame:
    X = data.drop(columns=[target])
    y = data[target]
    return X,y

# read parameters
def read_params(file_path: Path):
    with open(file_path, 'r') as f:
        params_file = yaml.safe_load(f)
    return params_file

# model fit
def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)
    return model

# save model 
def save_model(model, save_dir: Path, model_name: str):
    save_location = save_dir/model_name
    joblib.dump(value=model,filename=save_location)

# save model 
def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir/transformer_name
    joblib.dump(value=transformer,filename=save_location)

# Main block
if __name__=="__main__":
    # root path 
    root_path = Path(__file__).parent.parent.parent
    train_path = root_path/'data'/'processed'/'train_trans.csv'
    train_trans = load_data(train_path)
    logger.info("File load successfully")

    # split X_train and y_train
    X_train, y_train = split_X_and_y(data=train_trans, target=target_col)
    logger.info("Dataset splitting completed")

    # read parameters
    file_path = root_path/'params.yaml'
    model_params = read_params(file_path=file_path)['Train']
    # rf params
    rf_params = model_params['Random_Forest']
    logger.info("Random Forest Parameters Read")

    # build random forest model
    rf = RandomForestRegressor(**rf_params)
    logger.info("Built random forest model")

    # lightgbm params
    light_params = model_params['LightGBM']
    logger.info("Light GBM Parameters Read")

    # build light gbm model
    light = LGBMRegressor(**light_params)
    logger.info("Built Light GBM Model")  

    # meta model
    lr = LinearRegression()
    logger.info("Built Meta Model")

    # power_transfomer
    power_transformer = PowerTransformer()
    logger.info('Built Target Column Power Transformer')

    # form the stacking regressor
    stacking_reg = StackingRegressor(
        estimators=[
            ('rf_model', rf),
            ('light_model', light)
        ],
        final_estimator=lr,cv=5,n_jobs=-1
    )
    logger.info("Stacking Regressor Built")

    # make the model wraper
    model = TransformedTargetRegressor(
        regressor=stacking_reg,
        transformer=power_transformer
    )
    logger.info("Models Wrapped Inside")

    # fit the model on traing data
    train_model(model=model, X_train=X_train,y_train=y_train)
    logger.info("Model Traning Completed")

    # model name
    model_filename = 'model.joblib'
    model_save_dir = root_path/'models'
    model_save_dir.mkdir(exist_ok=True)

    # save the model
    save_model(model=model,
               save_dir = model_save_dir,
               model_name = model_filename)
    logger.info("Trained model saved to location")

    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save the stacking model
    stacking_filename = 'stacking_regressor.joblib'
    save_model(model=stacking_model,
               save_dir=model_save_dir,
               model_name=stacking_filename)
    logger.info("Stacking model save to location")
    
    # save the transformer
    transformer_filename = 'power_transformer.joblib'
    transformer_save_dir = model_save_dir
    save_transformer(transformer=transformer,
                     save_dir=transformer_save_dir,
                     transformer_name=transformer_filename
                    )
    logger.info("Transformer save to location")

