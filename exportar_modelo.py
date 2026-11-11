"""Momento 3: evaluación, optimización y persistencia del bosque de las prácticas."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from src.database.connection import MySQLConnector

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'resultados'


def main():
    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    # Conexión e ingesta desde la base de datos del proyecto de clase.
    db = MySQLConnector()
    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL
          AND TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) >= 18
        ORDER BY p.player_id
    """
    try:
        fecha = str(db.extract_data('SELECT CURDATE() AS fecha').iloc[0, 0])
        df = db.extract_data(query)
    finally:
        db.engine.dispose()

    # Preprocesamiento.
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')

    df = df.dropna(subset=['valor_mercado', 'edad']).copy()
    if df.empty or df.player_id.duplicated().any():
        raise ValueError('Se necesitan jugadores válidos con identificadores únicos.')

    # Definir la clase élite con el percentil 75, como en la guía.
    umbral_elite = float(df['valor_mercado'].quantile(0.75))
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    X = df[['edad', 'estatura_cm', 'posicion']].copy()
    y = df['es_elite']

    # Reservar jugadores para evaluar el aprendizaje en datos nuevos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    if y_train.value_counts().min() < 5:
        raise ValueError('Faltan ejemplos de ambas clases para cinco pliegues.')

    # Comparar los modelos solicitados para el análisis del informe.
    # pipeline() aplica las medianas y la codificación dentro de cada ajuste.
    modelos = {
        'Siempre estándar': pipeline(
            DummyClassifier(strategy='constant', constant=0)
        ),
        'Árbol libre': pipeline(
            DecisionTreeClassifier(max_depth=None, random_state=42)
        ),
        'Árbol podado': pipeline(
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
        ),
        'Árbol balanceado': pipeline(
            DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
        ),
        'Bosque de clase': pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=4, class_weight='balanced',
                random_state=42, n_jobs=2
            )
        ),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        print('Evaluando:', nombre, flush=True)
        modelo.fit(X_train, y_train)
        resultados[nombre] = evaluar(modelo, X_train, X_test, y_train, y_test)

    # Entrenar y optimizar el modelo final.
    malla = {
        'modelo__max_depth': [4, 6],
        'modelo__min_samples_leaf': [5, 10],
        'modelo__class_weight': ['balanced', None],
    }

    print("\nGridSearchCV: 8 combinaciones, 5 pliegues, 100 árboles por ajuste.", flush=True)
    busqueda = GridSearchCV(
        pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        malla,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=2,
        refit=True,
        error_score='raise',
    )
    busqueda.fit(X_train, y_train)

    optimo = busqueda.best_estimator_
    rf_model = optimo['modelo']
    modelos['Bosque optimizado'] = optimo
    resultados['Bosque optimizado'] = evaluar(
        optimo, X_train, X_test, y_train, y_test
    )

    # Guardar las medianas y categorías aprendidas para reutilizar en producción.
    imputador = optimo['preparar']['imputar']
    codificador = optimo['preparar']['columnas'].named_transformers_['posicion']
    columnas_X = optimo['preparar'].get_feature_names_out().tolist()
    medianas_estatura = imputador.alturas_.to_dict()
    mediana_global_estatura = float(imputador.altura_global_)
    mediana_global_edad = float(imputador.edad_global_)
    categorias_posicion = codificador.categories_[0].tolist()
    versiones = {p: version(p) for p in ['scikit-learn', 'pandas', 'numpy', 'joblib']}

    # Crear paquete de despliegue.
    paquete_despliegue = {
        'modelo': rf_model,
        'columnas_X': columnas_X,
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite,
        'categorias_posicion': categorias_posicion,
        'fecha_datos': fecha,
        'versiones': versiones,
    }

    (BASE / 'models').mkdir(exist_ok=True)
    ruta_modelo = BASE / 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo, compress=3)

    print("\n=================================================")
    print("  RESULTADO DE LA EVALUACIÓN DE MODELOS")
    print("=================================================")
    print(pd.DataFrame(resultados).T.round(4))
    print("=================================================\n")
    print("Mejores parámetros:", busqueda.best_params_)
    print(f"Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")

    # Guardar los resultados que respaldan las cifras del informe.
    importancia = pd.Series(
        rf_model.feature_importances_, index=columnas_X
    ).sort_values(ascending=False)

    resumen = {
        'fecha_datos': fecha,
        'ejecucion_utc': datetime.now(timezone.utc).isoformat(),
        'filas': len(df),
        'train': len(X_train),
        'test': len(X_test),
        'nulos_estatura': int(X.estatura_cm.isna().sum()),
        'clases': {str(k): int(n) for k, n in y.value_counts().sort_index().items()},
        'umbral_elite_eur': umbral_elite,
        'regla_elite': 'valor_mercado >= Q75 de la cohorte adulta',
        'mejores_parametros': busqueda.best_params_,
        'f1_cv': float(busqueda.best_score_),
        'metricas': resultados,
        'importancias': {k: float(n) for k, n in importancia.items()},
        'versiones': versiones,
        'metodologia': 'Cohorte de mayores de edad. Target Q75 global según la guía. División estratificada 80/20. Medianas y codificación aprendidas solo con Train y dentro de cada pliegue. Optimización por F1. Test reservado; el paquete conserva exactamente el modelo evaluado sin reentrenarlo con Test.',
    }

    SALIDA.mkdir(exist_ok=True)
    (SALIDA / 'metricas.json').write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    pd.DataFrame(resultados).T.to_csv(
        SALIDA / 'metricas.csv', encoding='utf-8-sig'
    )
    pd.DataFrame(busqueda.cv_results_)[[
        'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
    ]].to_csv(SALIDA / 'validacion_cruzada.csv', index=False)

    # Gráfica de entrenamiento y prueba para diagnosticar sobreajuste.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8, 4.3))
    etiquetas = ['Árbol libre', 'Árbol podado']
    x = np.arange(2)

    for desplazamiento, campo, etiqueta, color in [
        (-0.18, 'accuracy_train', 'Train', '#243b53'),
        (0.18, 'accuracy_test', 'Test', '#178a7b'),
    ]:
        barras = ax.bar(
            x + desplazamiento,
            [resultados[n][campo] for n in etiquetas],
            0.36, label=etiqueta, color=color
        )
        ax.bar_label(barras, fmt='%.3f', padding=3)

    ax.set(
        xticks=x, xticklabels=etiquetas, ylim=(0, 1),
        ylabel='Accuracy', title='Comparación de entrenamiento y prueba'
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / 'overfitting.png', dpi=200)
    plt.close(fig)

    # Curvas ROC del árbol y el bosque sobre los mismos datos de prueba.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for nombre, color in [
        ('Árbol balanceado', '#bd7730'),
        ('Bosque optimizado', '#178a7b'),
    ]:
        prob = modelos[nombre].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(
            fpr, tpr, color=color,
            label=f"{nombre} (AUC {resultados[nombre]['auc']:.3f})",
            linewidth=2
        )

    ax.plot([0, 1], [0, 1], ':', color='gray', label='Azar (AUC 0.500)')
    ax.set(
        xlabel='Tasa de falsos positivos',
        ylabel='Tasa de verdaderos positivos',
        title='Curvas ROC sobre la prueba reservada'
    )
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(SALIDA / 'roc.png', dpi=200)
    plt.close(fig)

    # Importancia de las variables utilizadas por el bosque optimizado.
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    importancia.sort_values().plot.barh(ax=ax, color='#178a7b')
    ax.set(
        xlabel='Importancia relativa por reducción de impureza',
        ylabel='',
        title='Variables utilizadas por el bosque optimizado'
    )
    fig.tight_layout()
    fig.savefig(SALIDA / 'importancias.png', dpi=200)
    plt.close(fig)


# Funciones de apoyo para la preparación y evaluación durante el entrenamiento.
# Se mantienen aquí para usar el mismo tratamiento en cada pliegue de validación.

class ImputadorPorPosicion(TransformerMixin, BaseEstimator):
    """Aprende las medianas dentro de cada pliegue de entrenamiento."""

    def _limpiar(self, X):
        datos = X[['edad', 'estatura_cm', 'posicion']].copy()
        for columna in ['edad', 'estatura_cm']:
            datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
        datos['posicion'] = datos['posicion'].fillna('Desconocida').astype(str)
        return datos

    def fit(self, X, y=None):
        datos = self._limpiar(X)
        self.alturas_ = datos.groupby('posicion')['estatura_cm'].median()
        self.altura_global_ = datos['estatura_cm'].median()
        self.edad_global_ = datos['edad'].median()
        if pd.isna(self.altura_global_) or pd.isna(self.edad_global_):
            raise ValueError('Entrenamiento sin edades o estaturas válidas para imputar.')
        self.feature_names_in_ = datos.columns.to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, 'alturas_')
        datos = self._limpiar(X)
        datos['estatura_cm'] = datos['estatura_cm'].fillna(
            datos['posicion'].map(self.alturas_)
        ).fillna(self.altura_global_)
        datos['edad'] = datos['edad'].fillna(self.edad_global_)
        return datos

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def crear_preprocesador():
    return Pipeline([
        ('imputar', ImputadorPorPosicion()),
        ('columnas', ColumnTransformer([
            ('numericas', 'passthrough', ['edad', 'estatura_cm']),
            ('posicion', OneHotEncoder(
                handle_unknown='ignore', drop='first', sparse_output=False
            ), ['posicion']),
        ], verbose_feature_names_out=False)),
    ])


def pipeline(estimador):
    return Pipeline([('preparar', crear_preprocesador()), ('modelo', estimador)])


def evaluar(modelo, X_train, X_test, y_train, y_test):
    pred = modelo.predict(X_test)
    scores = modelo.predict_proba(X_test)[:, list(modelo.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        'accuracy_train': float(accuracy_score(y_train, modelo.predict(X_train))),
        'accuracy_test': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, scores)),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


if __name__ == '__main__':
    main()
