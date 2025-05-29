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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import kaggle

def load_data():
    #pendiente modificar para que descargue de kaggle
    # df_1 = pd.read_csv(r'C:\Users\MdlSG\Documents\Master_II\_Dirección_organización_personas\HRClas\train_HRClas.csv')
    df_2 = pd.read_csv(r'.\hr_ana_act.csv')
    
    return df_2



def clarification_df(df):
    resources = df.keys()
    return resources

def preprocesar_dataframe(df: pd.DataFrame):
    # Elimina columna innecesaria
    
    # Columnas objetivo (por ejemplo, "se fue") debe separarse si estás entrenando
    if 'se fue' in df.columns:
        y = df['se fue'].values.astype(float)
        df = df.drop(columns=['se fue'])
    else:
        y = None
    
    # Variables categóricas -> One-hot encoding
    df = pd.get_dummies(df, columns=['departamento',  'región'])
    columnas_model=df.keys()
    

    # Escalado de datos numéricos (opcional pero recomendado)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Convertimos a tensores
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    return X_tensor, y_tensor, scaler, columnas_model


def visual_satis_zone(df_ana):
    
        
    st.title('Análisis del Nivel de Satisfacción')
    
    # Seleccionar la característica para el análisis
    feature = st.selectbox('Selecciona una característica', df_ana.columns.drop('nivel satisfacción'))
    if feature == 'horas mensuales promedio' or feature == 'última evaluación' or feature == 'salario':
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
def calc_renunciaAct(df):
          
    X, y, scaler, columnas_model = preprocesar_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RedNeuronalSimple(input_size=X_train.shape[1])
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)
    acc_epoch=[]
    for epoch in range(100):
        y_pred = modelo(X_train)
        loss = criterio(y_pred, y_train)
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        # Evaluación
        
        with torch.no_grad():
            y_pred_test = modelo(X_test)
            y_pred_labels = (y_pred_test >= 0.5).float()
            acc = accuracy_score(y_test.numpy(), y_pred_labels.numpy())
            acc_epoch.append(acc)
            # print(f"Precisión en test: {acc:.4f}")
    
    torch.save(modelo.state_dict(), "modelo.pth")
    
    return acc,acc_epoch,scaler,columnas_model
def calc_nuevo_renun(scaler,columnas_model):
     
    st.title("Formulario de Datos del Empleado")

    # Entradas numéricas
    nivel_satisfaccion = st.slider("Nivel de satisfacción (0 - 1)", 0.0, 1.0, 0.1)
    ultima_evaluacion = st.slider("Última evaluación (0 - 1)", 0.0, 1.0, 0.1)
    numero_proyectos = st.number_input("Número de proyectos", min_value=1, step=1)
    horas_mensuales = st.number_input("Horas mensuales promedio", min_value=200, step=10)
    anios_empresa = st.number_input("Años en la empresa", min_value=0, step=1)
    accidente_laboral = st.selectbox("¿Accidente laboral?", [0, 1])
    # se_fue = 0 # esto es el objetivo, realmente sería irrelevante pero por con
    promocion_5_anios = st.selectbox("¿Promoción en los últimos 5 años?", [0, 1])

    # Categóricos
    departamento = st.selectbox("Departamento", [
        'sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'
    ])
    salario = st.slider("Salario anual estimado", 25000, 75000, 45000, step=1000)
    region = st.selectbox("Región", ['Colombia', 'Reino Unido', 'India'])

    # Binarios adicionales
    expatriado = st.selectbox("¿Es expatriado?", [0, 1])
    ajuste_pci = st.selectbox("¿Recibió ajuste PCI?", [0, 1])
    conciliacion = st.selectbox("¿Tiene conciliación?", [0, 1])
    rotacion = st.selectbox("¿Alta rotación?", [0, 1])

    # Construcción del DataFrame con una fila
    data = {
        'nivel satisfacción': nivel_satisfaccion,
        'última evaluación': ultima_evaluacion,
        'número proyectos': numero_proyectos,
        'horas mensuales promedio': horas_mensuales,
        'años en la empresa': anios_empresa,
        'accidente laboral': accidente_laboral,
        # 'se fue': se_fue,
        'promoción últimos 5 años': promocion_5_anios,
        'departamento': departamento,
        'salario': salario,
        'región': region,
        'expatriado': expatriado,
        'ajuste PCI': ajuste_pci,
        'conciliacion': conciliacion,
        'rotacion': rotacion
    }

    df_usuario = pd.DataFrame([data])
    df_usuario=pd.get_dummies(df_usuario, columns=['departamento','región'])
    # Asegura que df_usuario tenga TODAS esas columnas:
    for col in columnas_model:
        if col not in df_usuario.columns:
            df_usuario[col] = 0  # agrega las columnas faltantes con 0

    # Ordena las columnas para que coincidan
    df_usuario = df_usuario[columnas_model]

    # Ahora ya puedes escalar
    X = scaler.transform(df_usuario.values)
    st.subheader("Datos ingresados")
    st.dataframe(df_usuario)
   
    X_tensor = torch.tensor(X, dtype=torch.float32)
    modelo = RedNeuronalSimple(input_size=X_tensor.shape[1])
    # modelo = RedNeuronalSimple(input_size=n)  # mismo input_size que en entrenamiento
    modelo.load_state_dict(torch.load("modelo.pth"))
    modelo.eval()
    prob=modelo(X_tensor)
   
    return prob
    
def truncar_float(f, n_decimales):
    factor = 10 ** n_decimales
    return int(f * factor) / factor


def main():
    st.header("ANÁLISIS RECURSOS HUMANOS")
    Texto= '¿Qué información desea consultar primero?'
    st.write(Texto)
    df_ana = load_data()
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
    if  checkbox_posible_exit:
        acc,acc_epoch,scaler,columnas_model=calc_renunciaAct(df_ana)
        
       
        x = (calc_nuevo_renun(scaler,columnas_model).item()*100)
        truncado =truncar_float(x, 2)
        st.write('La probabilidad de renuncia del supuesto empleado es de: \n'+ 
                 str(truncado)+' %')
        # # Gráfica
        # st.title("Precisión por época")
        # fig, ax = plt.subplots()
        # ax.plot(acc_epoch, label="Precisión")
        # ax.set_xlabel("Época")
        # ax.set_ylabel("Precisión")
        # ax.set_title("Evolución de la precisión en el conjunto de prueba")
        # ax.legend()
        # st.pyplot(fig)  
        # print (acc_epoch)
main()
#falta de comunicación
#falta de comunicacion de líderes, falta de acompañamiento
#apoyo instrumental del líder 
#reunione en grupo/ reuniones entre líderes => encuestas
#rotacion lideres entre zonas
#preparacion cultural de roles claves que vayan a ir a otros lugares
#cosas que no se pueden aplicar en paises, qué cosas funcionan en otros paises (a nivel cultural) (preguntar a DOMINGO)
#