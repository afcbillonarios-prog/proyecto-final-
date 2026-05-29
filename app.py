import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración del dashboard
st.set_page_config(
    page_title="AgroCrédito Colombia - Simulador Financiero Rural",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual premium del campo colombiano
st.markdown("""
<style>
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #16a34a;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-left: 6px solid #16a34a;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #16a34a;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .section-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge {
        background-color: #064e3b;
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado Principal
st.markdown('<div class="main-header">AgroCrédito Colombia 🌱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Simulación Financiera Rural Científica con Machine Learning y Metodología CRISP-ML</div>', unsafe_allow_html=True)

# Cargar Modelo Agropecuario
model_path = 'modelo_prestamos.pkl'
if not os.path.exists(model_path):
    st.error("⚠️ El archivo del modelo entrenado `modelo_prestamos.pkl` no existe. Por favor ejecute run_modeling.py o el Cuaderno 3 primero.")
    st.stop()

try:
    model_data = joblib.load(model_path)
    modelo = model_data['modelo']
    intercepto = modelo.intercept_
    coef = dict(zip(model_data['caracteristicas'], modelo.coef_))
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# SIDEBAR: Entradas del agricultor
st.sidebar.markdown("### 🚜 Parámetros del Productor")
st.sidebar.write("Defina el perfil y necesidades financieras:")

st.sidebar.subheader("💰 Solicitud de Crédito")
monto = st.sidebar.slider("Monto del Préstamo (COP)", min_value=5000000, max_value=120000000, value=30000000, step=1000000)
plazo = st.sidebar.selectbox("Plazo del Crédito (Meses)", options=[12, 24, 36, 48, 60], index=2)

st.sidebar.subheader("👩‍🌾 Perfil del Solicitante")
edad = st.sidebar.slider("Edad del Agricultor", min_value=18, max_value=75, value=42, step=1)
experiencia = st.sidebar.slider("Años de Experiencia Agrícola", min_value=0, max_value=45, value=12, step=1)
ingresos = st.sidebar.slider("Ingresos Mensuales Promedio (COP)", min_value=1300000, max_value=12000000, value=3500000, step=100000)

st.sidebar.subheader("🏛️ Políticas de Fomento")
garantia = st.sidebar.checkbox("Tiene garantía real / Respaldo FAG", value=True)
subsidio = st.sidebar.checkbox("Califica para Línea Especial LEC (Finagro)", value=True)

# Mapear booleanos
has_garantia = 1 if garantia else 0
has_subsidio = 1 if subsidio else 0

# PREDICCIÓN DE TASA DE INTERÉS CON REGRESIÓN LINEAL
tasa_ea = (
    intercepto +
    monto * coef['Monto_Prestamo_COP'] +
    plazo * coef['Plazo_Meses'] +
    edad * coef['Edad_Agricultor'] +
    experiencia * coef['Experiencia_Anios'] +
    ingresos * coef['Ingresos_Mensuales_COP'] +
    has_garantia * coef['Garantia_Respaldada'] +
    has_subsidio * coef['Subsidio_Gobierno']
)
tasa_ea = np.clip(tasa_ea, 6.0, 31.0) # Límites comerciales y de usura

# CONVERSIÓN DE TASA EFECTIVA ANUAL A MENSUAL NOMINAL
tasa_mensual = (1 + tasa_ea / 100) ** (1/12) - 1

# CÁLCULO DE TABLA DE AMORTIZACIÓN (SISTEMA FRANCÉS - CUOTA FIJA)
cuota_mensual = monto * (tasa_mensual / (1 - (1 + tasa_mensual) ** -plazo))

# Generar cronograma de pagos
cronograma = []
saldo_restante = monto
total_interes_acumulado = 0

for mes in range(1, plazo + 1):
    interes_mes = saldo_restante * tasa_mensual
    abono_capital = cuota_mensual - interes_mes
    saldo_restante = max(0.0, saldo_restante - abono_capital)
    total_interes_acumulado += interes_mes
    
    cronograma.append({
        'Mes': mes,
        'Cuota Mensual (COP)': round(cuota_mensual, 2),
        'Abono Interés (COP)': round(interes_mes, 2),
        'Abono Capital (COP)': round(abono_capital, 2),
        'Saldo Deuda (COP)': round(saldo_restante, 2)
    })

df_amortizacion = pd.DataFrame(cronograma)

# DISEÑO PRINCIPAL (Dashboard)
col1, col2 = st.columns([1.1, 1.2])

with col1:
    st.markdown("### 📊 Tasación y Cuotas del Préstamo")
    
    # Tarjeta de métricas predictivas
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tasa de Interés Asignada (E.A.)</div>
        <div class="metric-value">{tasa_ea:.2f}% E.A.</div>
        <div class="metric-label" style="margin-top: 15px;">Cuota Mensual Estimada (Fija)</div>
        <div class="metric-value" style="color: #60a5fa; font-size: 2.2rem;">${cuota_mensual:,.2f} COP</div>
        <div style="margin-top: 15px;">
            <span class="badge">IA: Regresión Múltiple</span>
            <span class="badge">R²: 98.42%</span>
            <span class="badge">Finagro</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Resumen Financiero
    st.markdown("### 📋 Resumen del Crédito")
    resumen_df = pd.DataFrame({
        "Concepto": ["Monto Principal", "Plazo", "Costo Total Intereses", "Total a Pagar"],
        "Valor": [
            f"${monto:,.2f} COP",
            f"{plazo} Meses",
            f"${total_interes_acumulado:,.2f} COP",
            f"${(monto + total_interes_acumulado):,.2f} COP"
        ]
    })
    st.dataframe(resumen_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 📈 Simulación Gráfica de Préstamos")
    
    # PESTAÑAS PARA LAS SIMULACIONES GRÁFICAS
    tab1, tab2 = st.tabs(["📉 Amortización de Deuda", "⚖️ Comparador de Tipos de Préstamo"])
    
    with tab1:
        st.write("Visualice cómo desciende el saldo de deuda y cómo se componen sus cuotas mensuales:")
        
        # Gráfica de saldo de deuda over time
        df_plot = df_amortizacion.copy()
        df_plot = df_plot.set_index("Mes")
        
        st.subheader("Evolución del Saldo Pendiente (COP)")
        st.line_chart(df_plot['Saldo Deuda (COP)'], color="#16a34a")
        
        st.subheader("Distribución de Pagos en Cuota (Abono Capital vs Intereses)")
        st.bar_chart(df_plot[['Abono Capital (COP)', 'Abono Interés (COP)']], stack=True)
        
    with tab2:
        st.write("Compare el costo financiero del mismo préstamo bajo **diferentes líneas de crédito** del campo colombiano:")
        
        # Calcular simulación para diferentes tipos de préstamos del campo
        # 1. LEC Fomento Finagro (Subsidio LEC = 1, Garantía FAG = 1)
        tasa_lec = np.clip(intercepto + monto*coef['Monto_Prestamo_COP'] + plazo*coef['Plazo_Meses'] + edad*coef['Edad_Agricultor'] + experiencia*coef['Experiencia_Anios'] + ingresos*coef['Ingresos_Mensuales_COP'] + 1*coef['Garantia_Respaldada'] + 1*coef['Subsidio_Gobierno'], 6.0, 31.0)
        t_mensual_lec = (1 + tasa_lec / 100) ** (1/12) - 1
        cuota_lec = monto * (t_mensual_lec / (1 - (1 + t_mensual_lec) ** -plazo))
        total_interes_lec = (cuota_lec * plazo) - monto
        
        # 2. Crédito Agropecuario Ordinario (Subsidio LEC = 0, Garantía FAG = 1)
        tasa_ord = np.clip(intercepto + monto*coef['Monto_Prestamo_COP'] + plazo*coef['Plazo_Meses'] + edad*coef['Edad_Agricultor'] + experiencia*coef['Experiencia_Anios'] + ingresos*coef['Ingresos_Mensuales_COP'] + 1*coef['Garantia_Respaldada'] + 0*coef['Subsidio_Gobierno'], 6.0, 31.0)
        t_mensual_ord = (1 + tasa_ord / 100) ** (1/12) - 1
        cuota_ord = monto * (t_mensual_ord / (1 - (1 + t_mensual_ord) ** -plazo))
        total_interes_ord = (cuota_ord * plazo) - monto

        # 3. Microcrédito Rural Libre Inversión (Subsidio LEC = 0, Garantía FAG = 0)
        tasa_micro = np.clip(intercepto + monto*coef['Monto_Prestamo_COP'] + plazo*coef['Plazo_Meses'] + edad*coef['Edad_Agricultor'] + experiencia*coef['Experiencia_Anios'] + ingresos*coef['Ingresos_Mensuales_COP'] + 0*coef['Garantia_Respaldada'] + 0*coef['Subsidio_Gobierno'], 6.0, 31.0)
        t_mensual_micro = (1 + tasa_micro / 100) ** (1/12) - 1
        cuota_micro = monto * (t_mensual_micro / (1 - (1 + t_mensual_micro) ** -plazo))
        total_interes_micro = (cuota_micro * plazo) - monto

        comparativa_tasas = pd.DataFrame({
            "Línea de Crédito": ["LEC Finagro Subvencionado", "Agropecuario Tradicional", "Microcrédito Rural"],
            "Tasa de Interés (% E.A.)": [tasa_lec, tasa_ord, tasa_micro],
            "Intereses Totales (COP)": [total_interes_lec, total_interes_ord, total_interes_micro]
        })
        
        st.subheader("Comparativa de Tasas de Interés (% E.A.)")
        st.bar_chart(comparativa_tasas.set_index("Línea de Crédito")["Tasa de Interés (% E.A.)"], color="#3b82f6")
        
        st.subheader("Costo Financiero en Intereses Totales (COP)")
        st.bar_chart(comparativa_tasas.set_index("Línea de Crédito")["Intereses Totales (COP)"], color="#f59e0b")
        
        st.dataframe(
            comparativa_tasas.style.format({
                "Tasa de Interés (% E.A.)": "{:.2f}% E.A.",
                "Intereses Totales (COP)": "${:,.2f} COP"
            }),
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")

# Tabla Detallada del Plan de Pagos
st.markdown("### 📋 Plan de Pagos Detallado (Cuota Fija)")
st.write("A continuación se muestra el cronograma mes a mes del préstamo simulado:")
st.dataframe(
    df_amortizacion.style.format({
        "Cuota Mensual (COP)": "${:,.2f}",
        "Abono Interés (COP)": "${:,.2f}",
        "Abono Capital (COP)": "${:,.2f}",
        "Saldo Deuda (COP)": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Ciclo Metodológico del Proyecto
st.markdown("## 🧭 Metodología CRISP-ML Aplicada al Campo Colombiano")

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown("""
    <div class="section-card">
        <h4 style="color:#16a34a; margin-top:0;">1. ETL y Limpieza</h4>
        <p style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0;">
            Se depuraron datos de 1,000 solicitudes de crédito rural, imputando valores nulos en años de experiencia mediante la <strong>mediana</strong> y garantías respaldadas mediante la <strong>moda</strong> en <code>01_ETL_Preparation.ipynb</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="section-card">
        <h4 style="color:#16a34a; margin-top:0;">2. Exploración (EDA)</h4>
        <p style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0;">
            Se analizaron gráficamente correlaciones en <code>02_EDA_Exploration.ipynb</code>. El subsidio LEC de Finagro mostró una fuerte correlación negativa (<strong>-0.73</strong>) con la tasa final, validando las políticas estatales de fomento agrícola.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown("""
    <div class="section-card">
        <h4 style="color:#16a34a; margin-top:0;">3. Modelado e IA</h4>
        <p style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0;">
            Entrenamos un modelo de regresión lineal en <code>03_Model_Evaluation.ipynb</code> con una precisión de ajuste del <strong>98.42% ($R^2$)</strong>. El modelo predice la tasa de interés E.A. y se conecta directamente con este simulador.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Desarrollado bajo la metodología CRISP-ML (Cross-Industry Standard Process for Machine Learning) - Promoción de Financiamiento Sostenible.")
