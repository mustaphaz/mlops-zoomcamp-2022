import os
import subprocess

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5001"
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
BEST_MODELEXPERIMENT_NAME = "random-forest-best-models"
ANSWERS = {}

mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def print_answers():
    for k, v in ANSWERS.items():
        print(f'{k} \n\t{v}')


def get_best_run_experiment(experiment_name):
    global best_run_hpo_experiment
    hpo_experiment = mlflow_client.get_experiment_by_name(experiment_name)
    best_run_hpo_experiment = mlflow_client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )
    return best_run_hpo_experiment[0]


if __name__ == '__main__':
    mlflow_version = subprocess.check_output("mlflow --version", shell=True).decode("utf-8")
    ANSWERS["Q1: What's the version of MLflow?"] = mlflow_version

    ANSWERS["Q2: How many files were saved to OUTPUT_FOLDER?"] = len(os.listdir('./output'))

    default_experiment = mlflow_client.get_experiment_by_name('Default')
    autolog_run = mlflow_client.search_runs(
        experiment_ids=default_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    ANSWERS["Q3: How many parameters are automatically logged by MLflow?"] = len(autolog_run[0].data.params)

    ANSWERS["Q4: In addition to backend-store-uri, what else do you need to pass to properly configure the server?"] = \
        "default-artifact-root"

    ANSWERS["Q5: What's the best validation RMSE that you got?"] = \
        get_best_run_experiment(HPO_EXPERIMENT_NAME).data.metrics['rmse']

    ANSWERS["Q6: What is the test RMSE of the best model?"] = \
        get_best_run_experiment(BEST_MODELEXPERIMENT_NAME).data.metrics['test_rmse']

    print_answers()
