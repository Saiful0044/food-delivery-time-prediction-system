import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = 'time_taken'

#  create logger
logger = logging.getLogger('data_preparation')
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

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.info("The file to load does not exits")
    return df 

# split data function
def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

# train test parameter read
def read_params(file_path: Path):
    with open(file_path,'r') as f:
        params_file = yaml.safe_load(f)
    return params_file

# train and test data save
def save_load(data: pd.DataFrame, save_path:Path)->None:
    data.to_csv(save_path, index=False) 


if __name__=='__main__':
    # root path
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path/'data'/'cleaned'/'cleaned_data.csv'
    data = load_data(data_path=data_path)
    logger.info("Data Loaded Successfully")

    # params yaml file path
    params_file_path = root_path/'params.yaml'
    parameters = read_params(params_file_path)['Data_Preparation']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logger.info("Parameters raed successfully")

    # split into train and test
    train_data, test_data = split_data(data=data,test_size=test_size, random_state=random_state)
    logger.info("Dataset split into train and test data")
    
    # save path for train and test
    save_data_dir = root_path/'data'/'interim'
    save_data_dir.mkdir(exist_ok=True, parents=True)

    train_file = 'train.csv'
    test_file = 'test.csv'
    save_train_path = save_data_dir/train_file
    save_test_path = save_data_dir/test_file

    data_subsets = [train_data,test_data]
    data_paths = [save_train_path,save_test_path]
    filename = [train_file,test_file]

    for file, path, data in zip(filename, data_paths,data_subsets):
        save_load(data=data, save_path=path)
        logger.info(f"{file.replace('.csv','')} data save to location")


    