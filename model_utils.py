import joblib
import pandas as pd
import numpy as np

def predecir_tasa_interes(
    monto_prestamo, 
    plazo_meses, 
    edad_agricultor, 
    experiencia_anios, 
    ingresos_mensuales, 
    garantia_respaldada, 
    subsidio_gobierno, 
    model_path='modelo_prestamos.pkl'
):
    """
    Carga el modelo de regresión lineal y predice la tasa de interés E.A. %
    para un crédito agropecuario en el campo colombiano.
    """
    try:
        # Cargar modelo
        data = joblib.load(model_path)
        modelo_cargado = data['modelo']
        columnas = data['caracteristicas']
        
        # DataFrame de entrada
        input_data = pd.DataFrame([[
            monto_prestamo, 
            plazo_meses, 
            edad_agricultor, 
            experiencia_anios, 
            ingresos_mensuales, 
            garantia_respaldada, 
            subsidio_gobierno
        ]], columns=columnas)
        
        # Predicción
        tasa_predicha = modelo_cargado.predict(input_data)[0]
        
        # Limitar en rangos válidos (6.0% a 31.0% de usura)
        tasa_final = np.clip(tasa_predicha, 6.0, 31.0)
        
        return round(float(tasa_final), 2)
    except Exception as e:
        print(f"Error al predecir tasa de interés: {e}")
        return None
