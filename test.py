import mlflow
import mlflow.xgboost  # XGBoost용 autolog 사용 (mlflow.sklearn 대신)
import xgboost as xgb  # XGBoost 라이브러리 임포트
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
# Iris 데이터셋의 타겟 변수는 이미 0, 1, 2 형태의 정수이므로 LabelEncoder는 필요 없습니다.

def main():
    # 1. MLflow 서버 URI 설정
    mlflow_server_uri = "http://127.0.0.1:5000"  # 로컬 서버 또는 실제 운영 서버 주소로 변경
    mlflow.set_tracking_uri(mlflow_server_uri)

    # 2. 실험(Experiment) 설정
    experiment_name = "XGBoost_Iris_Example"  # 실험 이름 변경
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        print(f"Could not set experiment {experiment_name}: {e}")

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    # 현재 활성화된 실험 정보를 가져오기 (선택 사항)
    try:
        active_experiment = mlflow.tracking.MlflowClient().get_experiment_by_name(experiment_name)
        if active_experiment:
            print(f"MLflow Experiment Name: {active_experiment.name}")
            print(f"MLflow Experiment ID: {active_experiment.experiment_id}")
    except Exception as e:
        print(f"Could not get experiment details: {e}")


    # 3. XGBoost 자동 로깅 활성화
    # mlflow.sklearn.autolog() 대신 mlflow.xgboost.autolog()를 사용합니다.
    # 이는 XGBoost 모델의 파라미터, 메트릭, 모델 자체를 자동으로 기록합니다.
    mlflow.xgboost.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)


    # 4. 데이터 로드 및 준비
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    # Iris 데이터의 타겟(y)은 이미 0, 1, 2로 되어 있어 XGBoost에 바로 사용 가능합니다.

    # 5. 모델 정의 및 학습 (XGBoost 사용)
    # XGBoost 하이퍼파라미터 예시
    params = {
        'objective': 'multi:softmax',  # 다중 클래스 분류 문제
        'num_class': 3,                # Iris 데이터셋의 클래스 개수
        'learning_rate': 0.08,
        'n_estimators': 150,           # 트리의 개수
        'max_depth': 6,                # 각 트리의 최대 깊이
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'eval_metric': 'mlogloss',     # 다중 분류용 평가 지표 (학습 중 로깅됨)
        'seed': 42
    }

    # autolog()를 사용하면 이 파라미터들이 자동으로 MLflow에 기록됩니다.
    model = xgb.XGBClassifier(**params)

    # XGBoost 모델 학습
    # autolog는 fit 과정에서 early stopping 관련 정보, 학습 과정의 평가 지표 등을 기록할 수 있습니다.
    # 더 많은 정보를 기록하려면 fit 메서드에 eval_set, early_stopping_rounds 등을 추가할 수 있습니다.
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], # 평가 데이터셋 지정
            )               # 학습 과정 출력 최소화


    # 6. 예측 및 평가
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # mlflow.xgboost.autolog()가 활성화되어 있으므로,
    # 정확도와 같은 최종 평가 메트릭은 자동으로 기록될 수 있습니다.
    # (만약 자동으로 기록되지 않는다면, mlflow.log_metric("test_accuracy", accuracy)로 수동 기록)

    current_run = mlflow.active_run()
    if current_run:
        print(f"Logging to MLflow Run ID: {current_run.info.run_id}")
    else:
        # `mlflow run`으로 실행 중이라면 이 메시지는 보통 나타나지 않습니다.
        # `python your_script.py`로 직접 실행하고 autolog가 run을 시작하지 않은 경우 발생 가능
        print("Warning: No active MLflow run detected by script logic after model training.")


    print(f"Run completed. Check MLflow UI for experiment '{experiment_name}'.")

if __name__ == "__main__":
    main()
