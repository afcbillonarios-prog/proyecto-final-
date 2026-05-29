import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def run_modeling():
    # Cargar datos limpios
    df = pd.read_csv('data/prestamos_campo_limpio.csv')
    
    # Definir variables predictoras y target
    caracteristicas = [
        'Monto_Prestamo_COP',
        'Plazo_Meses',
        'Edad_Agricultor',
        'Experiencia_Anios',
        'Ingresos_Mensuales_COP',
        'Garantia_Respaldada',
        'Subsidio_Gobierno'
    ]
    
    X = df[caracteristicas]
    y = df['Tasa_Interes_EA']
    
    # Partición train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Guardar modelo serializado
    model_data = {
        'modelo': modelo,
        'caracteristicas': caracteristicas
    }
    
    joblib.dump(model_data, 'modelo_prestamos.pkl')
    print("¡Modelo agrícola entrenado y guardado como 'modelo_prestamos.pkl' con éxito!")

if __name__ == '__main__':
    run_modeling()
