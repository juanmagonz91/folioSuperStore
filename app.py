import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib # To potentially save/load the model if needed, though caching is used here
import kagglehub # Import kagglehub to download the dataset

# Load the data
@st.cache_data
def load_data():
    # Download latest version of the dataset
    # Assuming the path is consistent with the previous notebook execution
    path = kagglehub.dataset_download("itssuru/super-store")

    # Display the path to the dataset files (optional in the final app, but useful for debugging)
    # st.text(f"Ruta a los archivos del dataset: {path}") # Moved to app.py

    df = pd.read_csv(os.path.join(path, 'SampleSuperstore.csv'))

    # Data cleaning and preparation
    df['Postal Code'] = df['Postal Code'].astype(str)
    categorical_cols = [
        'Ship Mode', 'Segment', 'Country', 'City',
        'State', 'Postal Code', 'Region', 'Category', 'Sub-Category'
    ]
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    df = df.drop_duplicates()
    df['is_unprofitable'] = (df['Profit'] < 0).astype(int)

    return df

df = load_data()

# Prepare data for modeling (done once)
# Select features. 'Country', 'City', 'State', 'Postal Code' are excluded due to high cardinality
# and less clear direct impact on profitability compared to the selected ones.
features = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount']

# One-hot encode categorical features and define X and y
X = pd.get_dummies(df[features], drop_first=True)
y = df['is_unprofitable']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the Gradient Boosting Classifier model (cached)
@st.cache_resource
def train_model(X_train, y_train):
    gbm_model = GradientBoostingClassifier(random_state=42)
    gbm_model.fit(X_train, y_train)
    return gbm_model

gbm_model = train_model(X_train, y_train)

# Evaluate the model and determine optimized threshold (cached)
@st.cache_data
def evaluate_model(_model, X_test, y_test):
    y_pred_proba = _model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Experiment with different thresholds to find an optimized one for recall
    thresholds = np.arange(0.1, 0.6, 0.05)
    best_recall = 0
    optimized_threshold = 0.5 # Default threshold

    for threshold in thresholds:
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        current_recall = recall_score(y_test, y_pred_adjusted)
        if current_recall > best_recall:
            best_recall = current_recall
            optimized_threshold = threshold

    return optimized_threshold, roc_auc

OPTIMIZED_THRESHOLD, roc_auc = evaluate_model(gbm_model, X_test, y_test)


st.title("Super Store Sales Analysis Dashboard")

# Sidebar for controls
st.sidebar.header("Panel de Control")

# Add exploratory data analysis widgets
st.sidebar.subheader("Análisis Exploratorio de Datos")
selected_region = st.sidebar.selectbox(
    "Seleccionar Región",
    options=['Todas'] + list(df['Region'].unique())
)

all_subcategories = list(df['Sub-Category'].unique())
selected_subcategories = st.sidebar.multiselect(
    "Seleccionar Subcategorías",
    options=all_subcategories,
    default=all_subcategories
)

max_discount = float(df['Discount'].max())
selected_discount_range = st.sidebar.slider(
    "Seleccionar Rango de Descuento",
    min_value=0.0,
    max_value=max_discount,
    value=(0.0, max_discount)
)

# Filter the DataFrame based on EDA selections
filtered_df = df.copy()

if selected_region != 'Todas':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

if selected_subcategories:
    filtered_df = filtered_df[filtered_df['Sub-Category'].isin(selected_subcategories)]

filtered_df = filtered_df[
    (filtered_df['Discount'] >= selected_discount_range[0]) &
    (filtered_df['Discount'] <= selected_discount_range[1])
]

# Main content area
st.header("Análisis Exploratorio de Datos")
st.markdown("Explora visualizaciones y resúmenes basados en tus selecciones en el Panel de Control.")

# Display EDA plots and tables (from previous step)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Ganancias")
    if not filtered_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        sns.histplot(filtered_df['Profit'], bins=50, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Histograma de Ganancias (Filtrado)')
        sns.boxplot(x=filtered_df['Profit'], ax=axes[1], color='lightgreen')
        axes[1].set_title('Boxplot de Ganancias (Filtrado)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("No hay datos disponibles para la distribución de Ganancias con los filtros actuales.")


with col2:
    st.subheader("Distribución por Categoría")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(y='Category', data=filtered_df, order=filtered_df['Category'].value_counts().index, palette='pastel', ax=ax)
        ax.set_title('Frecuencia por Categoría (Filtrado)')
        ax.set_xlabel('Cantidad')
        ax.set_ylabel('Categoría')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("No hay datos disponibles para la distribución por Categoría con los filtros actuales.")

st.subheader("Distribución por Subcategoría")
if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, max(5, len(filtered_df['Sub-Category'].unique()) * 0.5)))
    sns.countplot(y='Sub-Category', data=filtered_df, order=filtered_df['Sub-Category'].value_counts().index, palette='viridis', ax=ax)
    ax.set_title('Frecuencia por Subcategoría (Filtrado)')
    ax.set_xlabel('Cantidad')
    ax.set_ylabel('Subcategoría')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.write("No hay datos disponibles para la distribución por Subcategoría con los filtros actuales.")


st.subheader("Descuento vs. Ganancia (Filtrado)")
if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Discount', y='Profit', hue='Sub-Category', alpha=0.6, ax=ax)
    ax.set_title('Relación entre Descuento y Ganancia (Filtrado)')
    ax.set_xlabel('Descuento')
    ax.set_ylabel('Ganancia')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
else:
     st.write("No hay datos disponibles para el gráfico de dispersión Descuento vs. Ganancia con los filtros actuales.")


st.subheader("Tablas Resumen (Filtrado)")
if not filtered_df.empty:
    category_summary_filtered = filtered_df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    st.write("Ventas y Ganancias Totales por Categoría (Filtrado):")
    st.dataframe(category_summary_filtered)

    subcategory_summary_filtered = filtered_df.groupby('Sub-Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    st.write("Ventas y Ganancias Totales por Subcategoría (Filtrado):")
    st.dataframe(subcategory_summary_filtered)
else:
    st.write("No hay datos disponibles para las Tablas Resumen con los filtros actuales.")


# --- Predictive Analysis Section ---
st.header("Análisis Predictivo: Predecir Rentabilidad de Transacciones")
st.markdown("Utiliza el modelo para predecir si una transacción hipotética será no rentable.")

# Input widgets for a hypothetical transaction
st.subheader("Introduce los Detalles de la Transacción:")

col_pred_1, col_pred_2 = st.columns(2)

with col_pred_1:
    pred_ship_mode = st.selectbox("Modo de Envío", options=df['Ship Mode'].unique())
    pred_segment = st.selectbox("Segmento", options=df['Segment'].unique())
    pred_region = st.selectbox("Región", options=df['Region'].unique())
    pred_category = st.selectbox("Categoría", options=df['Category'].unique())

with col_pred_2:
    # Filter subcategories based on selected category
    filtered_subcategories = df[df['Category'] == pred_category]['Sub-Category'].unique()
    pred_sub_category = st.selectbox("Subcategoría", options=filtered_subcategories)
    pred_sales = st.number_input("Ventas", min_value=0.0, value=100.0)
    pred_quantity = st.number_input("Cantidad", min_value=1, value=2)
    pred_discount = st.slider("Descuento", min_value=0.0, max_value=max_discount, value=0.0, step=0.01)


# Button to trigger prediction
if st.button("Predecir Rentabilidad"):
    # Prepare input data for prediction
    input_data = {
        'Ship Mode': pred_ship_mode,
        'Segment': pred_segment,
        'Region': pred_region,
        'Category': pred_category,
        'Sub-Category': pred_sub_category,
        'Sales': pred_sales,
        'Quantity': pred_quantity,
        'Discount': pred_discount
    }
    input_df = pd.DataFrame([input_data])

    # One-hot encode the input data. Align with the training data columns.
    input_df_encoded = pd.get_dummies(input_df, columns=['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category'], drop_first=True)

    # Ensure the input DataFrame has the same columns as the training data X
    # Add missing columns (initialized to 0) and reorder columns
    missing_cols = set(X_train.columns) - set(input_df_encoded.columns)
    for c in missing_cols:
        input_df_encoded[c] = 0
    input_df_encoded = input_df_encoded[X_train.columns]


    # Get prediction probability
    pred_proba = gbm_model.predict_proba(input_df_encoded)[:, 1]
    unprofitable_proba = pred_proba[0]

    # Classify based on the optimized threshold
    predicted_class = "No Rentable" if unprofitable_proba >= OPTIMIZED_THRESHOLD else "Rentable"

    # Display the results
    st.subheader("Resultados de la Predicción:")
    st.write(f"Probabilidad Predicha de ser No Rentable: **{unprofitable_proba:.4f}**")
    st.write(f"Clasificación Predicha (usando umbral {OPTIMIZED_THRESHOLD:.2f}): **{predicted_class}**")

    if predicted_class == "No Rentable":
        st.warning("Se predice que esta transacción será no rentable.")
    else:
        st.success("Se predice que esta transacción será rentable.")


# Display Feature Importances
st.subheader("Importancia de las Características del Modelo")
st.markdown("Comprende qué factores el modelo considera más importantes para predecir la falta de rentabilidad.")

# Access feature importances from the trained model
feature_importances = gbm_model.feature_importances_
feature_importances_series = pd.Series(feature_importances, index=X_train.columns)
sorted_feature_importances = feature_importances_series.sort_values(ascending=False)

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=sorted_feature_importances.values, y=sorted_feature_importances.index, palette='viridis', ax=ax)
ax.set_title("Importancia de las Características para Predecir la Falta de Rentabilidad")
ax.set_xlabel("Puntuación de Importancia")
ax.set_ylabel("Característica")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Display Model Evaluation Metrics (Optional, based on previous analysis)
st.subheader("Métricas de Evaluación del Modelo")
st.markdown(f"El modelo fue evaluado con un umbral optimizado de {OPTIMIZED_THRESHOLD:.2f} para mejorar la identificación de transacciones no rentables.")

# Recalculate metrics with the optimized threshold for display
y_pred_optimized = (gbm_model.predict_proba(X_test)[:, 1] >= OPTIMIZED_THRESHOLD).astype(int)
accuracy_opt = accuracy_score(y_test, y_pred_optimized)
precision_opt = precision_score(y_test, y_pred_optimized)
recall_opt = recall_score(y_test, y_pred_optimized)
f1_opt = f1_score(y_test, y_pred_optimized)

st.write(f"**Exactitud:** {accuracy_opt:.4f}")
st.write(f"**Precisión:** {precision_opt:.4f}")
st.write(f"**Exhaustividad (Recall):** {recall_opt:.4f}")
st.write(f"**Puntuación F1:** {f1_opt:.4f}")
st.write(f"**AUC:** {roc_auc:.4f}")
