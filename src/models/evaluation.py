import mlflow.sklearn
import pandas as pd
import joblib
import logging
import mlflow
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
import dagshub

# initialize dagshub
dagshub.init(repo_owner='Saiful0044', repo_name='food-delivery-time-prediction-system', mlflow=True)


# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Saiful0044/food-delivery-time-prediction-system.mlflow")

# set the mlflow experment name
mlflow.set_experiment("DVC Pipeline v2")

TARGET = 'time_taken'

# create loggger
logger = logging.getLogger('model_evaluation')
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

# Data loading function
def load_data(data_path: Path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The data file not found")
        raise
    return df

# split data X,y
def split_X_and_y(data: pd.DataFrame, target: str):
    X = data.drop(columns=[target])
    y = data[target]
    return X,y

# model fuction
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# save mode info
def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict ={
        'run_id': run_id,
        'artifact_path': artifact_path,
        'model_name': model_name
    }
    with open(save_json_path, 'w') as f:
        json.dump(info_dict, f, indent=4)






# Main block 
if __name__=='_main__':
    # root path 
    root_path = Path(__file__).parent.parent.parent

    # train test data path load
    train_data_path = root_path / 'data'/'processed'/'train_trans.csv'
    test_data_path = root_path / 'data'/'processed'/'test_trans.csv'

    # load the training data
    train_data = load_data(data_path=train_data_path)
    logger.info("Train data loaded successfully")
    # load the testing data
    test_data = load_data(data_path=train_data_path)
    logger.info("Test data loaded successfully")

    # split the train and test data
    X_train,y_train = split_X_and_y(data=train_data, target=TARGET)
    X_test,y_test = split_X_and_y(data=test_data, target=TARGET)
    logger.info("Data split completed")

    # model path 
    model_path = root_path/'models'/'model.joblib'
    model = load_data(model_path=model_path)
    logger.info("Model Loaded Successfully")

    # get the predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction on data complete")

    # calculate the train and test mae
    train_mae = mean_absolute_error(y_train,y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    logger.info("Error Calculated")

    # calculate the r2 score
    train_r2 = r2_score(y_train,y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info("R2 Score Calculated")

    # calculate cross val scores
    cv_scores = cross_val_score(
        model,X_train,y_train,cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    logger.info('Cross Validation Complete')

    # mean cross val score
    mean_cv_score = -(cv_scores.mean())

    # log with mlflow
    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag('model', 'Food Delivery Time Regressor')

        # log Parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric('train_mae', train_mae)
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('train_r2', train_r2)
        mlflow.log_metric('test_r2', test_r2)
        mlflow.log_metric('mean_cv_score', mean_cv_score)

        # log individual cv scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,TARGET)

        #  log input
        mlflow.log_input(dataset=train_data_input, context='training')
        mlflow.log_input(dataset=test_data_input, context='validation')

        # model signature
        model_signature = mlflow.models.infer_signature(
            model_input=X_train.sample(20, random_state=42),
            model_output=model.predict(X_train.sample(20, random_state=42))
        )

        # log the final model
        mlflow.sklearn.log_model(model, 'delivery_time_pred_model', signature=model_signature)

        # log stacking regressor
        mlflow.log_artifact(root_path/'models'/'stacking_regressor.jobllib')

        # log the power transformer
        mlflow.log_artifact(root_path/'models'/'power_transformer.joblib')

        # log the preprocessor
        mlflow.log_artifact(root_path/'models'/'preprocessor.joblib')

        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()

        logger.info("Mlflow logging complete and model logged")

    # get the run id
    run_id = run.info.run_id
    model_name = 'delievery_time_pred_model'

    # save the model info
    save_json_path = root_path/'run_information.json'
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path = artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")



