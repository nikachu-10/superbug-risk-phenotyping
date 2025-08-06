# superbug-risk-phenotyping
Predicción del riesgo de infección y caracterización fenotípica de pacientes afectados por superbacterias en entornos hospitalarios mediante modelos de machine learning en Python

Este repositorio contiene el código desarrollado para un Trabajo de Fin de Máster, cuyo objetivo es predecir el riesgo de infección por superbacterias en pacientes hospitalarios e identificar los fenotipos clínicos más asociados a dichas infecciones, empleando técnicas de machine learning supervisado y no supervisado.

## Objetivos del proyecto
1. **Predecir el riesgo individual de infección por superbacterias**.
  - Predicción de la probabilidad de infección por **SARM** (*Staphylococcus aureus* resistente a meticilina) o *K*. BLEE (*Klebsiella pneumoniae* productora de beta-lactamasas de espectro extendido) a partir de características clínicas individuales de cada paciente.
  - Se aplican y evalúan múltiples modelos de clasificación.

2. **Identificar fenotipos clínicos asociados al riesgo de infección**.
   - Determinar perfiles de pacientes con mayor susceptibilidad a infecciones por las superbacterias estudiadas, mediante técnicas de agrupamiento (clustering) y modelos supervisados.

**Objetivos secundarios**:
    - **2.1**: Obtener agrupaciones homogéneas de pacientes utilizando algoritmos de clustering, evaluando si presentan diferentes riesgos de infección.
    - **2.2**: Extraer fenotipos clínicos a partir de las reglas aprendidas por los modelos de clasificación del objetivo 1.

# Datos utilizados
Se utilizaron dos **datasets de carácter privado** derivados de la base de datos **MIMIC-III**.
Incluyen información demográfica y clínica de pacientes hospitalarios completamente anonimizados.
**No están disponibles públicamente**.

# Modelos y técnicas utilizados
### Modelos de clasificación:
  - Regresión logística
  - SVM (kernel lineal)
  - Análisis discriminante lineal y cuadrático
  - Árbol de decisión
  - Random Forest
  - Extremely Random Trees (Extra Trees)
  - Gaussian Naive Bayes
  - K-vecinos más cercanos (KNN)
  - Gradient Boosting
  - XGBoost
  - AdaBoost
  - Voting Classifier

### Algoritmos de clustering:
  - K-means
  - Bisecting K-means
  - DBSCAN
  - HDBSCAN
  - Gaussian Mixture Models (GMM)

# Librerías empleadas
- `scikit-learn`  
- `xgboost`  
- `kneed`  
- `scipy`  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`

# Estructura del código
El proyecto se compone de 5 notebooks:

- `case_1_and_2.ipynb`: procesamiento inicial y preparación de datos
- `case_1_clasificacion.ipynb`.ipynb: clasificación y análisis de fenotipos para SARM
- `case_2_clasificacion.ipynb`: clasificación y análisis de fenotipos para Klebsiella BLEE
- `clustering_case_1.ipynb`: clustering de pacientes relacionados con SARM
- `clustering_case_2.ipynb`: clustering de pacientes relacionados con Klebsiella BLEE

# Requisitos de ejecución
  - **Python 3.12.7**
  - Entorno recomendado: **Visual Studio Code** (o cualquier IDE con soporte para Jupyter Notebooks).
Se recomienda el uso de un entorno virtual con las dependencias instaladas vía pip o requirements.txt (no incluido en este repo por contener librerías comunes).

# Resultados principales
Conclusiones para la predicción del riesgo individual:
  - **Modelo más eficaz para SARM**: Extra Trees, ajustado con 43 variables (AUC = 0.853)
  - **Modelo más eficaz para *K*. BLEE**: Voting Classifier, optimizado con 28 variables (AUC = 0.872)

Conclusiones del análisis fenotípico:
  - Existe una alta heterogeneidad en las muestras, lo que dificulta una clasificación clara mediante clustering. No se identificaron agrupamientos con alta correlación directa al estado infectado.
  - Características clínicas frecuentes en pacientes de alto riesgo:
    - **SARM**: trasplante de órganos previo, uso de catéteres, exposición previa a β-lactámicos y glicopéptidos.
    - ***K*. BLEE**: tiempo prolongado desde la admisión al ingreso en UCI, uso de catéteres, y exposición a múltiples antimicrobianos.

# Autor
Proyecto realizado como parte del Trabajo de Fin de Máster (TFM).
**Autor**: Nicolás Fernández Sobral
**Año**: 2025

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# superbug-risk-phenotyping

Risk prediction and phenotypic characterization of patients affected by superbug infections in hospital settings using machine learning models in Python.

This repository contains the code developed for a Master's Thesis project aimed at predicting the risk of superbug infections in hospitalized patients and identifying the clinical phenotypes most associated with such infections. The work applies both supervised and unsupervised machine learning techniques.

---

## Project Objectives

1. **Predict individual risk of superbug infection**
   - Predict the probability of infection by **MRSA** (Methicillin-resistant *Staphylococcus aureus*) or **ESBL-Klebsiella pneumoniae** based on individual clinical characteristics.
   - Multiple classification models are applied and evaluated.

2. **Identify clinical phenotypes associated with infection risk**
   - Determine profiles of patients with higher susceptibility to infection by the studied superbugs using clustering techniques and supervised learning.

   **Secondary objectives:**
   - **2.1**: Identify homogeneous patient clusters using clustering algorithms and assess whether these groups present different infection risks.
   - **2.2**: Extract clinical phenotypes based on the decision rules learned by the classification models developed in objective 1.

---

## Data Used

- Two **private datasets** derived from the **MIMIC-III** database were used.
- The datasets contain **anonymized demographic and clinical data** of hospitalized patients.
- The data is **not publicly available**.

---

## Models and Techniques

### Classification models:
- Logistic Regression  
- SVM (linear kernel)  
- Linear and Quadratic Discriminant Analysis  
- Decision Tree  
- Random Forest  
- Extremely Randomized Trees (Extra Trees)  
- Gaussian Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  
- XGBoost  
- AdaBoost  
- Voting Classifier

### Clustering algorithms:
- K-means  
- Bisecting K-means  
- DBSCAN  
- HDBSCAN  
- Gaussian Mixture Models (GMM)

---

## Libraries Used

- `scikit-learn`  
- `xgboost`  
- `kneed`  
- `scipy`  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`

---

## Code Structure

The project includes **5 Jupyter notebooks**:

- `case_1_and_2.ipynb`: initial data preprocessing and preparation  
- `case_1_clasificacion.ipynb`: classification and phenotype analysis for **MRSA**  
- `case_2_clasificacion.ipynb`: classification and phenotype analysis for **ESBL-Klebsiella pneumoniae**  
- `clustering_case_1.ipynb`: clustering of patients related to **MRSA**  
- `clustering_case_2.ipynb`: clustering of patients related to **Klebsiella BLEE**

---

## Execution Requirements

- **Python 3.12.7**
- Recommended IDE: **Visual Studio Code** (or any IDE supporting Jupyter Notebooks)
- It's recommended to use a virtual environment with dependencies installed via `pip`.  
  > Note: No `requirements.txt` is included as the project uses common libraries.

---

## Key Results

### Risk prediction:
- **Best model for MRSA**: Extra Trees (43 features, AUC = 0.853)
- **Best model for K. BLEE**: Voting Classifier (28 features, AUC = 0.872)

### Phenotype analysis:
- High heterogeneity among patients prevents straightforward classification using clustering.
- No strong correlation was found between clusters and infection status.
- Frequent high-risk clinical characteristics:
  - **MRSA**: prior organ transplant, catheter use, and prior exposure to β-lactams and glycopeptides.
  - **K. BLEE**: long time between hospital admission and ICU admission, catheter use, and high exposure to multiple antimicrobials.

---

## Author

This project was developed as part of a Master's Thesis.  
**Author:** Nicolás Fernández Sobral  
**Year:** 2025

