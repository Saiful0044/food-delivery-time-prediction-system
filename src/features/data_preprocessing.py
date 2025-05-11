import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn import set_config

num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]
target_col = "time_taken"


# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

# create logger
logger = logging.getLogger('data_preprocessing')
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

# drop missing value
def drop_missing_value(data:pd.DataFrame)-> pd.DataFrame:
    logger.info(f"The original dataset with missing values has {data.shape[0]} rows and {data.shape[1]} columns")
    df_dropped = data.dropna()
    logger.info(f"The dataset with missing values dropped has {df_dropped.shape[0]} row and {df_dropped.shape[1]} columns")
    missing_values = df_dropped.isna().sum().sum()
    
    if missing_values >0:
        raise ValueError("The dataframe has missing values")
    return df_dropped

# split data X and y
def make_X_and_y(data: pd.DataFrame, target:str)-> pd.DataFrame:
    X = data.drop(columns=[target])
    y = data[target]
    return X,y

# train preprocessor 
def train_preprocessor(preprocessor, data: pd.DataFrame)->pd.DataFrame:
    processing_data = preprocessor.fit(data)
    return processing_data


# perform transformation
def perform_transformation(preprocessor, data: pd.DataFrame) -> pd.DataFrame:
    transform_data = preprocessor.transform(data)
    return transform_data

# join x and y
def join_X_and_y(X,y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = y.reset_index(drop=True)
    return pd.concat([X,y], axis=1)

# transfomed save data 
def save_data(data: pd.DataFrame, save_path: Path):
    data.to_csv(save_path, index=False)


# save transfomer
def save_transfomer(transform, save_dir: Path, transform_name: str):
    save_location = save_dir/transform_name
    joblib.dump(value=transform, filename=save_location)




# Main Block 
if __name__=="__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path/'data'/'interim'/'train.csv'
    test_data_path = root_path/'data'/'interim'/'test.csv'
    train_data = drop_missing_value(load_data(train_data_path))
    logger.info('Train data loaded successfully')
    test_data = drop_missing_value(load_data(test_data_path))
    logger.info('Test data loaded successfully')

    # split data train,test
    X_train,y_train = make_X_and_y(data=train_data, target=target_col)
    X_test,y_test = make_X_and_y(data=test_data, target=target_col)
    logger.info('Data spliting completed')

    # preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('scaler', MinMaxScaler(), num_cols),
        ('nominal_cols', OneHotEncoder(drop='first', sparse_output=False,handle_unknown='ignore'), nominal_cat_cols),
        ('ordinal_cols', OrdinalEncoder(categories=[traffic_order,distance_type_order],encoded_missing_value=-999,handle_unknown="use_encoded_value",unknown_value=-1), ordinal_cat_cols)
    ], remainder='passthrough',n_jobs=-1)
    
    # fit the preprocessor on X_train
    train_preprocessor(preprocessor=preprocessor, data=X_train)
    logger.info("Preprocessor is trained")

    # transform the data
    X_train_trans = perform_transformation(preprocessor=preprocessor, data=X_train)
    logger.info("Train data is transformed")

    X_test_trans = perform_transformation(preprocessor=preprocessor, data=X_test)
    logger.info("Test data is transformed")

    # joni X and y
    train_trans_df = join_X_and_y(X_train_trans,y_train)
    test_trans_df = join_X_and_y(X_test_trans,y_test)
    logger.info('Datasets joined')

    # save the transformed data
    save_data_dir = root_path/'data'/'processed'
    save_data_dir.mkdir(exist_ok=True,parents=True)
    train_filename = 'train_trans.csv'
    test_filename = 'test_trans.csv'
    save_train_path = save_data_dir/train_filename
    save_test_path = save_data_dir/test_filename
    
    data_subsets = [train_trans_df,test_trans_df]
    data_paths = [save_train_path,save_test_path]
    filename_list = [train_filename,test_filename]
    for filename, path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data,save_path=path)
        logger.info(f"{filename.replace('.csv','')} data saved to location")

    transfomer_filename = 'preprocessor.joblib'
    transfomer_save_dir = root_path/'models'
    transfomer_save_dir.mkdir(exist_ok=True,parents=True)

    # Fix typo and filename
    save_transfomer(transform=preprocessor,
                    save_dir=transfomer_save_dir,
                    transform_name=transfomer_filename)
    logger.info("Preprocessor saved to location")


    
    

