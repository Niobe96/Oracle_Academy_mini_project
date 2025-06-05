import oracledb
import pandas as pd

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