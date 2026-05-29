import os
import numpy as np
import pandas as pd

def generate_dataset():
    np.random.seed(42)
    n_samples = 1000

    # Características del crédito del campo colombiano
    monto_prestamo = np.random.uniform(5000000, 120000000, n_samples).round(-3)  # En COP
    plazo_meses = np.random.choice([12, 24, 36, 48, 60], size=n_samples)
    edad_agricultor = np.random.randint(18, 76, n_samples)
    experiencia_anios = np.random.randint(0, 45, n_samples)
    ingresos_mensuales = np.random.uniform(1300000, 12000000, n_samples).round(-3)
    garantia_respaldada = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])  # FAG / Codeudor
    subsidio_gobierno = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])  # Finagro LEC

    # Tasa de Interés E.A. % (Target) con relación lineal razonable
    # Base: 25.0% E.A.
    # Subsidio Gobierno: -5.5%
    # Garantía FAG: -3.2%
    # Experiencia: -0.06% por año
    # Plazo: +0.03% por mes
    # Ingresos: -0.0000003 COP por peso ganado
    noise = np.random.normal(0, 0.8, n_samples)

    tasa_interes_ea = (
        25.0 -
        subsidio_gobierno * 5.5 -
        garantia_respaldada * 3.2 -
        experiencia_anios * 0.06 +
        plazo_meses * 0.03 -
        ingresos_mensuales * 0.0000003 +
        noise
    )

    # Limitar tasas de interés a rangos legales (mínimo 6.0% Finagro, máximo 31.0% de usura)
    tasa_interes_ea = np.clip(tasa_interes_ea, 6.0, 31.0).round(2)

    # Crear DataFrame
    df = pd.DataFrame({
        'ID_Prestamo': range(1, n_samples + 1),
        'Monto_Prestamo_COP': monto_prestamo,
        'Plazo_Meses': plazo_meses,
        'Edad_Agricultor': edad_agricultor,
        'Experiencia_Anios': experiencia_anios,
        'Ingresos_Mensuales_COP': ingresos_mensuales,
        'Garantia_Respaldada': garantia_respaldada,
        'Subsidio_Gobierno': subsidio_gobierno,
        'Tasa_Interes_EA': tasa_interes_ea
    })

    # Introducir algunos nulos para la etapa de ETL
    # 2% de nulos en 'Experiencia_Anios' y 1.5% en 'Garantia_Respaldada'
    mask_experiencia = np.random.rand(n_samples) < 0.02
    mask_garantia = np.random.rand(n_samples) < 0.015
    df.loc[mask_experiencia, 'Experiencia_Anios'] = np.nan
    df.loc[mask_garantia, 'Garantia_Respaldada'] = np.nan

    # Crear carpeta data si no existe
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/prestamos_campo.csv', index=False)
    print("Dataset prestamos_campo.csv generado exitosamente en carpeta 'data/'!")

if __name__ == '__main__':
    generate_dataset()
