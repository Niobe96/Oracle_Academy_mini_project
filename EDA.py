import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import seaborn as sns # 현재 코드에서 seaborn 직접 사용 부분 없음
# import matplotlib.pyplot as plt # 현재 코드에서 matplotlib 직접 사용 부분 없음

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, \
    LabelEncoder as SkLabelEncoder  # Sklearn의 LabelEncoder도 사용 가능
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="고급 CSV 데이터 분석기")

# --- 타이틀 ---
st.title("🛠️ 고급 CSV 데이터셋 분석기")
st.markdown("""
CSV 파일을 업로드하고 데이터 개요 확인, Feature Engineering, 간단한 모델 비교를 수행할 수 있습니다.
사이드바에서 옵션을 선택하여 분석을 진행하세요.
""")


# --- 데이터 로드 함수 (캐싱 사용) ---
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df.copy()  # 원본 수정을 방지하기 위해 복사본 반환
    except Exception as e:
        st.error(f"파일을 읽는 중 오류 발생: {e}")
        return None


# --- 세션 상태 초기화 ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:  # 원본 데이터 보존용
    st.session_state.original_df = None
if 'ordinal_maps' not in st.session_state:  # 순서형 인코딩 맵 저장용
    st.session_state.ordinal_maps = {}

# --- 사이드바 ---
st.sidebar.header("⚙️ 분석 옵션")

# --- 1. 파일 업로드 ---
uploaded_file = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None or uploaded_file.name != st.session_state.get('uploaded_file_name'):
        loaded_df = load_data(uploaded_file)
        if loaded_df is not None:
            st.session_state.original_df = loaded_df.copy()
            st.session_state.df = loaded_df.copy()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.ordinal_maps = {}  # 새 파일 로드 시 맵 초기화
            st.sidebar.success("파일이 성공적으로 업로드 및 초기화되었습니다!")
        else:
            st.session_state.df = None
            st.session_state.original_df = None

if st.session_state.df is not None:
    df = st.session_state.df

    st.header("📄 데이터 개요")
    if st.sidebar.checkbox("데이터 미리보기 (상위 5개 행)", True):
        st.subheader("데이터 미리보기 (Head)")
        st.dataframe(df.head())

    if st.sidebar.checkbox("데이터프레임 정보 (Shape, Dtypes)", False):
        st.subheader("데이터프레임 정보")
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    if st.sidebar.checkbox("기술 통계량", False):
        st.subheader("기술 통계량 (Numerical Columns)")
        st.dataframe(df.describe(include=np.number))
        st.subheader("기술 통계량 (Categorical Columns)")
        st.dataframe(df.describe(include='object'))

    st.header("🛠️ Feature Engineering")
    st.sidebar.subheader("Feature Engineering 옵션")

    if st.sidebar.button("Feature Engineering 초기화 \n (원본 데이터로 복원)"):
        if st.session_state.original_df is not None:
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.ordinal_maps = {}  # 맵도 초기화
            df = st.session_state.df
            st.success("데이터가 원본 상태로 복원되었습니다.")
        else:
            st.warning("원본 데이터가 없습니다. 파일을 다시 업로드해주세요.")

    # --- 3.1 범주형 데이터 인코딩 ---
    st.sidebar.markdown("#### 범주형 데이터 인코딩")
    encoding_method = st.sidebar.radio(
        "인코딩 방법 선택",
        ("선택 안 함", "원-핫 인코딩", "순서형 인코딩 (Label Encoding)"),
        key="encoding_method_selector"
    )

    categorical_cols_for_encoding = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if encoding_method == "원-핫 인코딩":
        st.subheader("원-핫 인코딩 (One-Hot Encoding)")
        if not categorical_cols_for_encoding:
            st.info("원-핫 인코딩을 적용할 범주형 컬럼이 없습니다.")
        else:
            selected_ohe_cols = st.multiselect("원-핫 인코딩을 적용할 범주형 컬럼 선택", categorical_cols_for_encoding, key="ohe_cols")
            if selected_ohe_cols:
                if st.button("원-핫 인코딩 적용", key="apply_ohe"):
                    try:
                        # 이미 인코딩된 컬럼은 제외 (예: _ohe, _ordinal 접미사가 붙은 경우)
                        actual_ohe_cols = [col for col in selected_ohe_cols if
                                           not (col.endswith("_ohe") or col.endswith("_ordinal"))]
                        if not actual_ohe_cols:
                            st.warning("선택된 컬럼 중 실제 원본 범주형 컬럼이 없거나 이미 변환된 컬럼일 수 있습니다.")
                        else:
                            df_ohe = pd.get_dummies(df, columns=actual_ohe_cols, prefix=actual_ohe_cols, dummy_na=False)
                            st.session_state.df = df_ohe
                            df = st.session_state.df
                            st.success(f"{actual_ohe_cols} 컬럼에 대한 원-핫 인코딩이 적용되었습니다.")
                            st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"원-핫 인코딩 중 오류 발생: {e}")

    elif encoding_method == "순서형 인코딩 (Label Encoding)":
        st.subheader("순서형 인코딩 (Label Encoding)")
        if not categorical_cols_for_encoding:
            st.info("순서형 인코딩을 적용할 범주형 컬럼이 없습니다.")
        else:
            selected_ordinal_col = st.selectbox("순서형 인코딩을 적용할 범주형 컬럼 선택", categorical_cols_for_encoding,
                                                key="ordinal_col")
            if selected_ordinal_col:
                # 이미 변환된 컬럼인지 확인 (예: _ordinal 접미사)
                if selected_ordinal_col.endswith("_ordinal") or selected_ordinal_col.endswith("_ohe"):
                    st.warning(f"'{selected_ordinal_col}' 컬럼은 이미 변환된 컬럼일 수 있습니다. 원본 범주형 컬럼을 선택해주세요.")
                elif st.button(f"'{selected_ordinal_col}'에 순서형 인코딩 적용", key="apply_ordinal"):
                    try:
                        # 고유값 등장 순서대로 매핑 (1부터 시작)
                        unique_values = df[selected_ordinal_col].astype(
                            'category').cat.categories  # .unique()는 순서 보장 안 할 수 있음. category로 변환 후 categories 사용
                        if not list(unique_values):  # unique()가 빈 경우 등 예외처리
                            unique_values = df[selected_ordinal_col].unique()

                        ordinal_map = {value: i + 1 for i, value in enumerate(unique_values)}

                        new_col_name = f"{selected_ordinal_col}_ordinal"
                        df[new_col_name] = df[selected_ordinal_col].map(ordinal_map)

                        st.session_state.df = df
                        st.session_state.ordinal_maps[new_col_name] = ordinal_map  # 생성된 맵 저장
                        df = st.session_state.df

                        st.success(f"'{selected_ordinal_col}' 컬럼에 순서형 인코딩이 적용되어 '{new_col_name}' 컬럼이 생성되었습니다.")
                        st.write("적용된 매핑:")
                        st.json(ordinal_map)
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"순서형 인코딩 중 오류 발생: {e}")

    # 저장된 순서형 인코딩 맵 표시
    if st.session_state.ordinal_maps:
        with st.expander("현재 적용된 순서형 인코딩 맵 보기"):
            for col_name, a_map in st.session_state.ordinal_maps.items():
                st.write(f"**{col_name}**:")
                st.json(a_map)

    # --- 3.2 수치형 변수 변환 ---
    st.sidebar.markdown("#### 수치형 변수 변환")
    if st.sidebar.checkbox("수치형 변수 변환 활성화", False, key="enable_num_transform"):
        st.subheader("수치형 변수 변환")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        # 이미 변환된 컬럼은 제외 (예: _poly, _log, _sqrt, _ordinal, _ohe가 붙은 것들)
        numerical_cols = [col for col in numerical_cols if
                          not any(suffix in col for suffix in ['_poly', '_log', '_sqrt', '_ordinal', '_ohe'])]

        if not numerical_cols:
            st.info("변환을 적용할 원본 수치형 컬럼이 없습니다.")
        else:
            selected_num_col_trans = st.selectbox("변환을 적용할 수치형 컬럼 선택", numerical_cols, key="num_col_trans")
            transformation_type = st.selectbox("변환 종류 선택",
                                               ["다항 변수 생성", "로그 변환 (log1p)", "제곱근 변환"],
                                               key="trans_type")

            if selected_num_col_trans:  # 컬럼이 선택되었는지 확인
                if transformation_type == "다항 변수 생성":
                    degree = st.number_input("다항식 차수", min_value=2, max_value=5, value=2, key="poly_degree")
                    if st.button(f"'{selected_num_col_trans}'에 다항 변수 적용 (차수: {degree})", key="apply_poly"):
                        try:
                            poly = PolynomialFeatures(degree=degree, include_bias=False)
                            # 해당 컬럼만 추출하여 2D 배열로 만듦
                            col_data = df[[selected_num_col_trans]].copy()
                            # NaN 값 처리 후 변환
                            imputer = SimpleImputer(strategy='mean')
                            col_data_imputed = pd.DataFrame(imputer.fit_transform(col_data), columns=col_data.columns,
                                                            index=col_data.index)

                            poly_features = poly.fit_transform(col_data_imputed)

                            # 원본 컬럼을 제외한 새로운 다항식 항들의 이름 생성 (x^2, x^3, ...)
                            # poly.get_feature_names_out()는 입력 컬럼 이름을 기반으로 이름을 생성
                            # 예: 입력이 'col_name'이면, 'col_name', 'col_name^2', ...
                            feature_names_out = poly.get_feature_names_out([selected_num_col_trans])

                            new_poly_cols_added = []
                            for i, feature_name in enumerate(feature_names_out):
                                if f"^{i + 1}" not in feature_name and i > 0:  # 원본 컬럼(i=0)은 제외, x^2부터 추가
                                    new_col_name_df = f"{selected_num_col_trans}_poly{i + 1}"  # get_feature_names_out이 생성한 이름 사용
                                    if feature_name == selected_num_col_trans: continue  # 원본 컬럼은 건너뛰기
                                    df[new_col_name_df] = poly_features[:, i]
                                    new_poly_cols_added.append(new_col_name_df)
                                elif f"^{i + 1}" in feature_name and selected_num_col_trans in feature_name:  # x^2, x^3 같은 경우
                                    power = int(feature_name.split('^')[-1])
                                    if power > 1:  # 1차항(원본)은 추가하지 않음
                                        new_col_name_df = f"{selected_num_col_trans}_poly{power}"
                                        df[new_col_name_df] = poly_features[:, i]
                                        new_poly_cols_added.append(new_col_name_df)

                            if new_poly_cols_added:
                                st.session_state.df = df
                                st.success(f"다항 변수가 추가되었습니다: {', '.join(new_poly_cols_added)}")
                                st.dataframe(df.head())
                            else:
                                st.warning("새로운 다항 변수가 추가되지 않았습니다. 차수나 컬럼을 확인해주세요.")
                        except Exception as e:
                            st.error(f"다항 변수 생성 중 오류: {e}")


                elif transformation_type == "로그 변환 (log1p)":
                    if st.button(f"'{selected_num_col_trans}'에 로그 변환 적용", key="apply_log"):
                        try:
                            new_log_col_name = f"{selected_num_col_trans}_log"
                            # np.log1p는 log(1+x)를 계산하여 x가 0이어도 안전. 음수 값은 처리 필요.
                            col_data = df[selected_num_col_trans].copy()
                            if (col_data < 0).any():
                                st.warning(
                                    f"'{selected_num_col_trans}'에 음수가 포함되어 있습니다. 로그 변환 전 절대값을 취하거나, 양수로 조정해야 합니다. 여기서는 (x - min(x)) + 1 에 log1p를 적용합니다.")
                                min_val = col_data.min()
                                col_data_adjusted = col_data - min_val  # 모든 값을 0 이상으로 만듦
                            else:
                                col_data_adjusted = col_data

                            df[new_log_col_name] = np.log1p(col_data_adjusted)
                            st.session_state.df = df
                            st.success(f"로그 변환된 컬럼 '{new_log_col_name}'이 추가되었습니다.")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"로그 변환 중 오류: {e}")


                elif transformation_type == "제곱근 변환":
                    if st.button(f"'{selected_num_col_trans}'에 제곱근 변환 적용", key="apply_sqrt"):
                        try:
                            new_sqrt_col_name = f"{selected_num_col_trans}_sqrt"
                            col_data = df[selected_num_col_trans].copy()
                            if (col_data < 0).any():
                                df[new_sqrt_col_name] = np.sqrt(np.abs(col_data))
                                st.warning(f"'{selected_num_col_trans}'에 음수가 있어 np.sqrt(abs(x))를 적용했습니다.")
                            else:
                                df[new_sqrt_col_name] = np.sqrt(col_data)
                            st.session_state.df = df
                            st.success(f"제곱근 변환된 컬럼 '{new_sqrt_col_name}'이 추가되었습니다.")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"제곱근 변환 중 오류: {e}")

    # --- 3.3 상호작용 항 생성 ---
    st.sidebar.markdown("#### 상호작용 항 생성")
    if st.sidebar.checkbox("상호작용 항 생성 활성화", False, key="enable_interaction"):
        st.subheader("상호작용 항 생성 (두 수치형 변수의 곱)")
        numerical_cols_interaction = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols_interaction = [col for col in numerical_cols_interaction if not any(
            suffix in col for suffix in ['_poly', '_log', '_sqrt', '_ordinal', '_ohe'])]

        if len(numerical_cols_interaction) < 2:
            st.info("상호작용 항을 생성하려면 최소 2개의 원본 수치형 컬럼이 필요합니다.")
        else:
            col1_interact = st.selectbox("첫 번째 수치형 컬럼 선택", numerical_cols_interaction, key="interact_col1")
            available_cols2_interaction = [col for col in numerical_cols_interaction if col != col1_interact]

            if not available_cols2_interaction:
                st.warning("두 번째로 선택할 수 있는 수치형 컬럼이 없습니다.")
            else:
                col2_interact = st.selectbox("두 번째 수치형 컬럼 선택", available_cols2_interaction, key="interact_col2")
                if col1_interact and col2_interact:  # 두 컬럼 모두 선택되었는지 확인
                    if st.button(f"'{col1_interact}' * '{col2_interact}' 상호작용 항 추가", key="apply_interaction"):
                        try:
                            new_interaction_col_name = f"{col1_interact}_x_{col2_interact}"
                            df[new_interaction_col_name] = df[col1_interact] * df[col2_interact]
                            st.session_state.df = df
                            st.success(f"상호작용 항 '{new_interaction_col_name}'이 추가되었습니다.")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"상호작용 항 생성 중 오류: {e}")

    # --- 4. 기본적인 모델 비교 프레임워크 ---
    st.header("🤖 기본적인 모델 비교 프레임워크")
    st.sidebar.subheader("모델 비교 옵션")

    if st.sidebar.checkbox("모델 비교 실행", False, key="enable_model_comparison"):
        st.subheader("모델 비교 설정")

        available_cols_for_model = df.columns.tolist()

        if not available_cols_for_model:
            st.warning("모델 학습에 사용할 수 있는 컬럼이 없습니다. 데이터를 먼저 로드하고 Feature Engineering을 수행하세요.")
        else:
            target_variable = st.selectbox("목표 변수 (Target Variable) 선택", available_cols_for_model, key="target_var",
                                           index=None, placeholder="목표 변수를 선택하세요...")

            if target_variable:
                feature_variables = st.multiselect("특성 변수 (Feature Variables) 선택",
                                                   [col for col in available_cols_for_model if col != target_variable],
                                                   key="feature_vars", placeholder="특성 변수들을 선택하세요...")

                if feature_variables:
                    try:
                        X = df[feature_variables].copy()  # 경고 방지를 위해 .copy() 사용
                        y = df[target_variable].copy()

                        # NaN 값 처리 (X, y 모두)
                        if X.isnull().values.any():
                            st.warning("선택된 특성에 NaN 값이 포함되어 있습니다. 평균값으로 대체합니다.")
                            imputer_X = SimpleImputer(strategy='mean')
                            X_imputed = imputer_X.fit_transform(X)
                            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

                        if y.isnull().values.any():
                            st.warning(f"목표 변수 '{target_variable}'에 NaN 값이 포함되어 있습니다. 해당 행을 제외합니다.")
                            # y에서 NaN이 있는 행의 인덱스를 가져와 X와 y에서 모두 제거
                            nan_indices_y = y[y.isnull()].index
                            y = y.drop(nan_indices_y)
                            X = X.drop(nan_indices_y)
                            if y.empty or X.empty:
                                st.error("NaN 값 제거 후 데이터가 남아있지 않아 모델 학습을 진행할 수 없습니다.")
                                st.stop()

                        # 수치형 특성 스케일링 (X 전체에 대해)
                        numerical_features_in_X = X.select_dtypes(include=np.number).columns
                        if len(numerical_features_in_X) > 0:
                            scaler = StandardScaler()
                            X.loc[:, numerical_features_in_X] = scaler.fit_transform(X[numerical_features_in_X])

                        task_type = None
                        y_unique_count = y.nunique()
                        y_dtype_is_numeric = pd.api.types.is_numeric_dtype(y)

                        if y_dtype_is_numeric and y_unique_count > 10:  # 수치형이면서 고유값 10개 초과시 회귀 (임계값 조절 가능)
                            task_type = "Regression"
                        elif y_unique_count <= 10:  # 고유값 10개 이하면 분류로 간주 (수치형이라도)
                            task_type = "Classification"
                            if y_unique_count == 2 and (
                                    pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(
                                    y) or y_dtype_is_numeric):
                                # 이진 분류의 경우, 값이 숫자가 아니거나 0,1이 아니면 레이블 인코딩
                                if not (y.isin([0, 1]).all() and y_dtype_is_numeric):
                                    le = SkLabelEncoder()  # sklearn의 LabelEncoder 사용
                                    y = le.fit_transform(y)
                                    y = pd.Series(y, index=X.index)  # 인덱스 맞춰주기
                                    st.info(
                                        f"목표 변수 '{target_variable}'가 이진 분류를 위해 레이블 인코딩되었습니다. (0, 1) 원본 값 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                            elif y_unique_count > 2 and (
                                    pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(
                                    y) or y_dtype_is_numeric):
                                # 다중 클래스 분류 시에도 레이블 인코딩 (0, 1, 2...)
                                le = SkLabelEncoder()
                                y = le.fit_transform(y)
                                y = pd.Series(y, index=X.index)  # 인덱스 맞춰주기
                                st.info(
                                    f"목표 변수 '{target_variable}'가 다중 클래스 분류를 위해 레이블 인코딩되었습니다. 원본 값 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                        else:
                            st.error(
                                f"목표 변수 '{target_variable}'의 유형을 결정할 수 없습니다. (고유값: {y_unique_count}, 데이터타입: {y.dtype})")
                            st.stop()

                        st.write(f"**감지된 문제 유형**: {task_type}")

                        if st.button("선택한 변수로 모델 비교 실행", key="run_model_comparison"):
                            if X.empty or y.empty:
                                st.error("특성(X) 또는 목표(y) 데이터가 비어있어 모델 학습을 진행할 수 없습니다.")
                                st.stop()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                                stratify=y if task_type == "Classification" and y.nunique() > 1 else None)

                            models = {}
                            results = []

                            with st.spinner("모델을 학습하고 평가하는 중입니다..."):
                                if task_type == "Regression":
                                    models_config = {
                                        "Linear Regression": LinearRegression(),
                                        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                                        "Random Forest Regressor": RandomForestRegressor(random_state=42,
                                                                                         n_estimators=50)
                                    }
                                    for name, model_instance in models_config.items():
                                        model_instance.fit(X_train, y_train)
                                        y_pred = model_instance.predict(X_test)
                                        mse = mean_squared_error(y_test, y_pred)
                                        r2 = r2_score(y_test, y_pred)
                                        results.append({"Model": name, "MSE": f"{mse:.4f}", "R-squared": f"{r2:.4f}"})

                                elif task_type == "Classification":
                                    models_config = {
                                        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200,
                                                                                  solver='liblinear'),
                                        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                                        "Random Forest Classifier": RandomForestClassifier(random_state=42,
                                                                                           n_estimators=50)
                                    }
                                    y_test_nunique = y_test.nunique()  # y_test의 고유값으로 average 결정

                                    for name, model_instance in models_config.items():
                                        model_instance.fit(X_train, y_train)
                                        y_pred = model_instance.predict(X_test)
                                        accuracy = accuracy_score(y_test, y_pred)

                                        f1_average_method = 'binary' if y_test_nunique == 2 else 'weighted'
                                        f1 = f1_score(y_test, y_pred, average=f1_average_method, zero_division=0)

                                        roc_auc = "N/A"
                                        if hasattr(model_instance, "predict_proba"):
                                            y_pred_proba = model_instance.predict_proba(X_test)
                                            if y_test_nunique == 2:  # 이진 분류
                                                try:
                                                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                                                    roc_auc = f"{roc_auc:.4f}"
                                                except ValueError as e_roc:  # y_test에 하나의 클래스만 있는 경우 등
                                                    roc_auc = f"계산 불가 ({e_roc})"
                                            # 다중 클래스 ROC AUC는 생략
                                        results.append(
                                            {"Model": name, "Accuracy": f"{accuracy:.4f}", "F1 Score": f"{f1:.4f}",
                                             "ROC AUC": roc_auc})

                            if results:
                                results_df = pd.DataFrame(results)
                                st.subheader("모델 비교 결과")
                                st.dataframe(results_df.set_index("Model"))
                            else:
                                st.warning("모델 학습 결과를 가져올 수 없습니다.")
                    except Exception as e_model:
                        st.error(f"모델 비교 중 오류 발생: {e_model}")
                        st.error(f"오류 타입: {type(e_model).__name__}, 오류 메시지: {str(e_model)}")
                        import traceback

                        st.text(traceback.format_exc())

    st.sidebar.markdown("---")
    st.sidebar.info("모든 분석 옵션은 데이터셋의 특성에 따라 결과가 다를 수 있습니다.")

elif uploaded_file is not None and st.session_state.df is None:
    st.error("데이터프레임을 로드하지 못했습니다. 파일 형식을 확인하거나 다른 파일을 시도해 보세요.")

else:
    st.info("분석을 시작하려면 사이드바에서 CSV 파일을 업로드하세요.")
    st.sidebar.info("파일을 업로드하면 분석 옵션이 활성화됩니다.")

st.markdown("---")
st.caption("Streamlit CSV 분석기 by Gemini")