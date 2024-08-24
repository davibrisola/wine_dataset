import numpy as np
import pandas as pd
import sklearn.datasets


def safe_to_datetime(month, day):
    try:
        return pd.to_datetime(f'1990-{month:02d}-{day:02d}', format="%Y-%m-%d")
    except ValueError:
        return pd.NaT  # Retorna um valor nulo para datas inválidas
    
    
# Carregar o conjunto de dados
data = sklearn.datasets.load_wine()

# Criar o DataFrame com os dados
df = pd.DataFrame(data.data, columns=data.feature_names)

# Selecionar as colunas relevantes
df = df[['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity']]

# Definir a semente para resultados reprodutíveis
np.random.seed(1)

# Gerar colunas para dia e mês aleatórios
day = np.random.choice(30, len(df))
mon = np.random.choice([8, 9, 10], len(df))

# Adicionar coluna de data ao DataFrame
df['date_measure'] = [safe_to_datetime(mon[i], day[i]) for i in range(len(df))]

# Definir a coluna de data como índice e ordenar
df = df.set_index('date_measure').sort_index()

    
def get_mean_magnesium(df):
    return df['magnesium'].mean()

def get_min_alcohol_september(df):
    return df[df.index.month == 9]['alcohol'].min()

def get_observations_count(df):
    return df[(df['color_intensity'] > 8.0) & (df['malic_acid'] < 20)].shape[0]

def get_month_with_highest_std(df):
    df['ratio'] = df['malic_acid'] / df['ash']
    monthly_std = df.groupby(df.index.month)['ratio'].std()
    return monthly_std.idxmax()

def get_date_with_min_diff(df):
    df['flavanoids_max'] = df.groupby(df.index.date)['flavanoids'].transform('max')
    df['flavanoids_diff'] = df.groupby(df.index.date)['flavanoids_max'].diff().abs()
    
    # Obter o índice da menor diferença
    min_diff_idx = df['flavanoids_diff'].idxmin()
    
    # Acessar a data correspondente
    date_value = df.loc[min_diff_idx, 'flavanoids_diff']
    
    return min_diff_idx.strftime('%Y-%m-%d')

# Armazenamento para melhorar visualização dos dados
mean_magnesium = get_mean_magnesium(df)
min_alcohol_september = get_min_alcohol_september(df)
observations_count = get_observations_count(df)
month_highest_std = get_month_with_highest_std(df)
date_min_diff = get_date_with_min_diff(df)

q1 = mean_magnesium
q2 = min_alcohol_september
q3 = observations_count
q4 = month_highest_std
q5 = date_min_diff

if __name__ == "__main__":
    print(q1)
    print(q2)
    print(q3)
    print(q4)
    print(q5)