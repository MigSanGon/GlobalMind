import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# import kaggle

def load_data():
    #pendiente modificar para que descargue de kaggle
    # df_1 = pd.read_csv(r'C:\Users\MdlSG\Documents\Master_II\_Dirección_organización_personas\HRClas\train_HRClas.csv')
    df_2 = pd.read_csv(r'.\hr_ana_act.csv')
    
    return df_2



def clarification_df(df):
    resources = df.keys()
    return resources
def visual_satis_zone(df_ana):
    
        
    st.title('Análisis del Nivel de Satisfacción')
    
    # Seleccionar la característica para el análisis
    feature = st.selectbox('Selecciona una característica', df_ana.columns.drop('nivel satisfacción'))
    if feature == 'horas mensuales promedio' or feature == 'última evaluación':
        min_value = df_ana[feature].min()
        max_value = df_ana[feature].max()
        range_values = st.slider('Selecciona el rango de valores', min_value, max_value, (min_value, max_value))
        
        filtered_data = df_ana[(df_ana[feature] >= range_values[0]) & (df_ana[feature] <= range_values[1])]
    else:
         filtered_data = df_ana

    # Calcular la satisfacción media por cada valor de la característica seleccionada
    mean_satisfaction = filtered_data.groupby(feature)['nivel satisfacción'].mean().reset_index()

    # Crear gráfica de barras
    fig, ax = plt.subplots()
    ax.bar(mean_satisfaction[feature], mean_satisfaction['nivel satisfacción'])
    ax.set_xlabel(feature)
    ax.set_ylabel('Satisfacción Media')
    ax.set_title(f'Satisfacción Media por {feature}')

    # Mostrar gráfica
    st.pyplot(fig)


    return 

def calc_renunciaAct(df):
    sali= df['se fue'].sum() / len(df) * 100
    total= df.count()
    class RedNeuronalSimple(nn.Module):
        def __init__(self, input_size):
            super(RedNeuronalSimple, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 8),  # Capa oculta con 8 neuronas
                nn.ReLU(),
                nn.Linear(8, 1),           # Capa de salida (1 neurona para clasificación binaria)
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    X = torch.tensor(df.values, dtype=torch.float32)
    y = torch.tensor(df['se fue'], dtype=torch.float32).unsqueeze(1)

    modelo = RedNeuronalSimple(input_size=X.shape[1])
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)

    # Entrenamiento simplificado
    for epoch in range(100):
        y_pred = modelo(X)
        loss = criterio(y_pred, y)
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()


def main():
    st.header("ANÁLISIS RECURSOS HUMANOS")
    Texto= '¿Qué información desea consultar primero?'
    st.write(Texto)
    df_clas,df_ana = load_data()
    col1, col2  = st.columns(2)
    if "checkbox_r" not in st.session_state:
            st.session_state.checkbox_r = False
    if "checkbox_c" not in st.session_state:
            st.session_state.checkbox_c = False
    # Añadir checkboxes a cada columna
    with col1:
        checkbox_satisf = st.checkbox("Satisfacción empleados", value = False)
    with col2:
        checkbox_posible_exit = st.checkbox("Posibles renuncias", value = False)
    if checkbox_satisf:
        visual_satis_zone(df_ana)

main()
#falta de comunicación
#falta de comunicacion de líderes, falta de acompañamiento
#apoyo instrumental del líder 
#reunione en grupo/ reuniones entre líderes => encuestas
#rotacion lideres entre zonas
#preparacion cultural de roles claves que vayan a ir a otros lugares
#cosas que no se pueden aplicar en paises, qué cosas funcionan en otros paises (a nivel cultural) (preguntar a DOMINGO)
#