import pandas as pd

def run_etl():
    # Cargar dataset
    df = pd.read_csv('data/prestamos_campo.csv')
    
    # Imputar valores nulos
    # Experiencia_Anios: Imputar con la mediana
    mediana_experiencia = df['Experiencia_Anios'].median()
    df['Experiencia_Anios'] = df['Experiencia_Anios'].fillna(mediana_experiencia)
    
    # Garantia_Respaldada: Imputar con la moda (0 o 1)
    moda_garantia = df['Garantia_Respaldada'].mode()[0]
    df['Garantia_Respaldada'] = df['Garantia_Respaldada'].fillna(moda_garantia)
    
    # Conversión de tipos de datos
    df['Plazo_Meses'] = df['Plazo_Meses'].astype(int)
    df['Edad_Agricultor'] = df['Edad_Agricultor'].astype(int)
    df['Experiencia_Anios'] = df['Experiencia_Anios'].astype(int)
    df['Garantia_Respaldada'] = df['Garantia_Respaldada'].astype(int)
    df['Subsidio_Gobierno'] = df['Subsidio_Gobierno'].astype(int)
    
    # Guardar dataset limpio
    df.to_csv('data/prestamos_campo_limpio.csv', index=False)
    print("¡Archivo data/prestamos_campo_limpio.csv creado con éxito en ETL!")

if __name__ == '__main__':
    run_etl()
