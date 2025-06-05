import oracledb
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_23_8")

user = "EDU107"
password = "oracle_4U"
host = '138.2.63.245'
port = 1521
service ="srvinv.sub03250142080.kdtvcn.oraclevcn.com"
dsn = f"{host}:{port}/{service}"

connection = oracledb.connect(
    user=user,
    password=password,
    dsn=dsn,
)

query_apache = "SELECT * FROM kdtuser.KDT_SEPSIS_CRF_APACHE"
query_sofa = "SELECT * FROM kdtuser.KDT_SEPSIS_CRF_SOFA"

df_apache = pd.read_sql(query_apache, con=connection)
df_sofa = pd.read_sql(query_sofa, con=connection)

df_apache_sorted = df_apache.sort_values(by="ID").reset_index(drop=True)
df_sofa_sorted = df_sofa.sort_values(by="ID").reset_index(drop=True)

df_merged = pd.concat([df_apache_sorted, df_sofa_sorted], axis=1)
df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
print(f"전처리 전 입력변수: {df_merged.shape}")

df_clean = df_merged.drop(columns=["ID", "Adm", "Death_time", "Tr_time", "endtime"])
X = df_clean.dropna()
print(f"전처리 후 입력변수: {X.shape}")

y = X['result'].apply(lambda x: 0 if x in [0,1] else 1)
print(f"label: {y.shape}")
X = X.drop(columns=["result"])
num_columns = X.shape[1]

count_0 = (y==0).sum()
count_1 = (y==1).sum()
print(f"y = 0인 경우의 개수: {count_0}")
print(f"y = 1인 경우의 개수: {count_1}")

if count_0 >= 30 and count_1 >= 30:
    X_features = X
    group_0 = X_features[y == 0]
    group_1 = X_features[y == 1]
    
    categorical_cols = [
        'liver', 'ARF', 'multiorgan', 'Immunocompromised', 'Op', 'Renal', 'Heart',
        'sum', 'Respiratory', 'Sex', 'VASO', 'MV', 'AF_init', 'AF_hos'
    ]
    
    continous_results = []
    categorical_results = []
    
    for col in X_features.columns:
        if group_0[col].isnull().values.any() or group_1[col].isnull().values.any():
            continue

if col in categorical_cols:
    count_0_val = group_0[col].value_counts().sotrt_index()
    count_1_val = group_1[col].value_counts().sort_index()  
    total_0 = count_0_val.sum()
    total_1 = count_1_val.sum()
    
    precent_0 = count_0_val / total_0 * 100
    percnet_1 = count_1_val / total_1 * 100
    
    total_count = X_features[col].value_counts().sort_index()
    total_total = total_count.sum()
    total_percent = total_count / total_total * 100 
    
    contingency = pd.corsstab(y, X_features[col])
    try:
        chi2, p_value, _, _ = chi2_contingency(contingency)
    except:
        p_value = np.nan
        
    categorical_results.append({
        'Feature' : col,
        'Total_Counts' : dict(total_count),
        'Total_Percent' : dict(total_percent.round(2)),
        'Survive_Counts' : dict(count_0_val),
        'Survive_Percent' : dict(precent_0.round(2)),
        'Death_Counts' : dict(count_1_val),
        'Death_Percent' : dict(percnet_1.round(2)),
        'p-value' : p_value
    })
    
else:
    mean_0 = group_0[col].mean()
    mean_1 = group_1[col].mean()
    std_0 = group_0[col].std()
    std_1 = group_1[col].std()
    
    overall_mean = X_features[col].mean()
    overall_std = X_features[col].std()
    
    t_stat, p_value = ttest_ind(group_0[col], group_1[col], equal_var=False)
    
    continous_results.append({
        'Feature' : col,
        'Total_Mean' : round(overall_mean, 2),
        'Total_Std' : round(overall_std, 2),
        'Survive_Mean' : round(mean_0, 2),
        'Survive_Std' : round(std_0, 2),
        'Death_Mean' : round(mean_1, 2),
        'Death_Std' : round(std_1, 2),
        'p-value' : p_value
    })
    
df_continous_stats = pd.DataFrame(continous_results)
df_categorical_stats = pd.DataFrame(categorical_results)

print(df_continous_stats)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


learning_rate = 0.002

def build_model(num_input =1, lr=learning_rate):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=num_input))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = build_model(num_input=X_train.shape[1], lr=learning_rate)
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath ='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.25,
    callbacks=[early_stopping, model_checkpoint],
    verbose=2
)

def plot_loss_curve(history, start=1):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    total_epoch = len(train_loss)
    
    if start < 1:
        start = 1
    if start > total_epoch:
        print(f"시작 epoch {start}가 학습된 epoch 수{total_epoch}보다 큽니다")
        return
    
    epochs_range = range(start, total_epoch + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs_range, train_loss[start-1:], label='Training Loss')
    plt.plot(epochs_range, val_loss[start-1:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss Curves')
    plt.show()

plot_loss_curve(history)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"test accuracy: {test_acc:.4f}")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color='darkorange', lw=2)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()




shap.initjs()
explainer = shap.DeepExplainer(model, X_train.values)
shap_values = explainer.shap_values(X_test.values)


plt.figure()
shap.summary_plot(shap_values[1], X_test.values, feature_names=X.columns, show=True)

plt.figure()
shap.summary_plot(shap_values[1], X_test.values, feature_names=X.columns, plot_type="bar",show=True)
plt.show()
