"""app_crediticio.py
Plataforma Integral de Análisis Crediticio y Asesoría Financiera.

Requisitos de instalación:
    pip install gradio pandas numpy matplotlib scikit-learn xgboost google-generativeai

Variable de entorno obligatoria para Tab 2:
    GEMINI_API_KEY — clave de la API de Google Gemini
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE: bool = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from google import genai
    from google.genai.types import GenerateContentConfig

    GEMINI_AVAILABLE: bool = True
except ImportError:
    GEMINI_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de la API de Gemini
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────────────────────
# Clase ModelTrainer
# ──────────────────────────────────────────────────────────────────────────────


class ModelTrainer:
    """Pipeline de modelado estadístico para análisis de riesgo crediticio.

    Gestiona la carga de datos, preprocesamiento, entrenamiento de múltiples
    modelos de clasificación y el cálculo de indicadores de discriminación
    utilizados en scorecard crediticio (Gini, KS, AUC-ROC, Odds Ratio).

    Attributes:
        df: Dataset cargado por el usuario.
        columns: Nombres de columnas del dataset activo.
        roc_data: Lista de dicts con datos para renderizar curvas ROC por modelo.

    Example:
        >>> trainer = ModelTrainer()
        >>> preview, cols, msg = trainer.load_data("datos.csv")
        >>> metrics_df, fig = trainer.train_and_evaluate("target", ["edad", "ingresos"])
    """

    def __init__(self) -> None:
        """Inicializa ModelTrainer con atributos de estado vacíos."""
        self.df: Optional[pd.DataFrame] = None
        self.columns: List[str] = []
        self.roc_data: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_path(file: Any) -> str:
        """Extrae la ruta del objeto de archivo retornado por gr.File.

        Args:
            file: Objeto retornado por gr.File (puede ser str, NamedString,
                  TemporaryFileWrapper, etc., dependiendo de la versión de Gradio).

        Returns:
            Ruta absoluta del archivo temporal como cadena.
        """
        if isinstance(file, str):
            return file
        if hasattr(file, "name"):
            return str(file.name)
        return str(file)

    def load_data(
        self, file: Any
    ) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """Carga un archivo CSV o TXT y devuelve una vista previa de 10 filas.

        Intenta múltiples separadores para archivos TXT. Actualiza los atributos
        internos ``df`` y ``columns`` si la carga es exitosa.

        Args:
            file: Archivo subido a través del componente gr.File.

        Returns:
            Tupla con:
                - DataFrame con las primeras 10 filas (None si hay error).
                - Lista de nombres de columnas disponibles.
                - Mensaje de estado descriptivo para mostrar al usuario.

        Raises:
            gr.Error: Si el formato no es soportado, el archivo está vacío,
                      tiene menos de 2 columnas o no puede ser parseado.
        """
        if file is None:
            return None, [], "⚠️ No se ha cargado ningún archivo."

        try:
            path = self._resolve_path(file)
            ext = os.path.splitext(path)[1].lower()

            if ext == ".csv":
                df = pd.read_csv(path)

            elif ext == ".txt":
                df = None
                for sep in ["\t", ",", ";", "|"]:
                    try:
                        candidate = pd.read_csv(path, sep=sep)
                        if candidate.shape[1] > 1:
                            df = candidate
                            break
                    except Exception:
                        continue
                if df is None:
                    raise ValueError(
                        "No se pudo determinar el separador del archivo .txt. "
                        "Formatos aceptados: tabulación, coma, punto y coma, pipe."
                    )
            else:
                raise ValueError(
                    f"Formato '{ext}' no soportado. Cargue un archivo .csv o .txt."
                )

            if df is None or df.empty:
                raise ValueError("El archivo no contiene datos.")
            if df.shape[1] < 2:
                raise ValueError(
                    "El dataset debe tener al menos 2 columnas (target + 1 feature)."
                )

            self.df = df
            self.columns = df.columns.tolist()
            status = (
                f"✅ Datos cargados exitosamente — "
                f"{df.shape[0]:,} filas × {df.shape[1]} columnas."
            )
            return df.head(10), self.columns, status

        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"Error al cargar el archivo: {exc}")

    # ── Indicadores de discriminación ──────────────────────────────────────────

    @staticmethod
    def _compute_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcula el coeficiente de Gini a partir del AUC-ROC.

        El Gini mide la capacidad discriminante del modelo; valores más altos
        indican mejor separación entre buenos y malos pagadores.

        Args:
            y_true: Etiquetas binarias reales (0 = bueno, 1 = malo).
            y_score: Probabilidades predichas para la clase positiva.

        Returns:
            Gini = 2 · AUC − 1, en el rango [-1, 1].
        """
        return 2.0 * float(roc_auc_score(y_true, y_score)) - 1.0

    @staticmethod
    def _compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcula el estadístico KS (Kolmogorov-Smirnov).

        Representa la máxima distancia vertical entre las distribuciones
        acumuladas de buenos y malos pagadores a lo largo de todos los umbrales.

        Args:
            y_true: Etiquetas binarias reales.
            y_score: Probabilidades predichas para la clase positiva.

        Returns:
            KS = max(TPR − FPR), en el rango [0, 1].
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(np.abs(tpr - fpr)))

    @staticmethod
    def _compute_odds_ratio(pipeline: Pipeline) -> str:
        """Extrae el odds ratio promedio de un paso LogisticRegression.

        El odds ratio exp(β) indica el cambio multiplicativo en las probabilidades
        de impago por unidad de incremento en cada variable predictora.

        Args:
            pipeline: Pipeline de sklearn entrenado con un paso denominado 'model'.

        Returns:
            Odds ratio promedio formateado con 4 decimales, o 'N/A' si el modelo
            no es una regresión logística.
        """
        try:
            mdl = pipeline.named_steps.get("model")
            if isinstance(mdl, LogisticRegression):
                mean_or = float(np.mean(np.exp(mdl.coef_[0])))
                return f"{mean_or:.4f}"
        except Exception:
            pass
        return "N/A"

    # ── Visualización ──────────────────────────────────────────────────────────

    def _build_roc_figure(self) -> plt.Figure:
        """Construye una figura matplotlib con las curvas ROC de todos los modelos.

        Usa los datos almacenados en ``self.roc_data`` para superponer las curvas
        de cada modelo entrenado sobre una línea base de clasificador aleatorio.

        Returns:
            Figura de matplotlib lista para renderizar en gr.Plot.
        """
        palette = ["#2563EB", "#16A34A", "#DC2626", "#7C3AED"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        for i, entry in enumerate(self.roc_data):
            ax.plot(
                entry["fpr"],
                entry["tpr"],
                color=palette[i % len(palette)],
                linewidth=2.4,
                label=f"{entry['name']}  (AUC = {entry['auc']:.4f})",
            )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.4,
            alpha=0.50,
            label="Clasificador aleatorio  (AUC = 0.50)",
        )

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.06])
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
        ax.set_title(
            "Curvas ROC — Comparativa de Modelos de Riesgo Crediticio",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.set_facecolor("#F9FAFB")
        fig.patch.set_facecolor("#FFFFFF")
        plt.tight_layout()
        return fig

    # ── Entrenamiento y evaluación ─────────────────────────────────────────────

    def train_and_evaluate(
        self,
        target: str,
        features: List[str],
        test_size: float = 0.25,
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """Entrena y compara Regresión Logística, Random Forest y XGBoost.

        Realiza partición estratificada entrenamiento/prueba, codificación de
        variables categóricas y cálculo de los indicadores de scorecard:
        AUC-ROC, Gini, KS y Odds Ratio.

        Args:
            target: Nombre de la columna objetivo binaria (0 / 1).
            features: Lista de columnas predictoras a incluir.
            test_size: Fracción del dataset reservada para prueba (default: 0.25).

        Returns:
            Tupla con:
                - DataFrame con métricas comparativas por modelo.
                - Figura matplotlib con las curvas ROC superpuestas.

        Raises:
            gr.Error: Si no hay datos cargados, la selección es inválida,
                      el target no es binario o el dataset tiene menos de 50 filas.
        """
        if self.df is None:
            raise gr.Error(
                "Primero debe cargar un dataset mediante la sección 'Carga de Datos'."
            )
        if not target:
            raise gr.Error("Debe seleccionar la variable objetivo (Target).")
        if not features:
            raise gr.Error(
                "Debe seleccionar al menos una variable explicativa (Feature)."
            )
        if target in features:
            raise gr.Error(
                "La variable objetivo no puede incluirse entre las variables explicativas."
            )

        try:
            subset = self.df[features + [target]].copy()

            n_before = len(subset)
            subset = subset.dropna()
            n_dropped = n_before - len(subset)
            if n_dropped:
                gr.Warning(
                    f"Se eliminaron {n_dropped} filas con valores nulos "
                    f"({n_dropped / n_before * 100:.1f}% del total)."
                )

            if len(subset) < 50:
                raise gr.Error(
                    f"El dataset válido contiene sólo {len(subset)} filas. "
                    "Se necesitan al menos 50 registros para entrenar los modelos."
                )

            # Codificación de variables categóricas
            X = subset[features].copy()
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            y = subset[target]
            if y.nunique() != 2:
                raise gr.Error(
                    f"La variable objetivo '{target}' tiene {y.nunique()} valores únicos. "
                    "Debe ser estrictamente binaria (exactamente 2 clases: 0 y 1)."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Definición de pipelines
            model_pipelines: Dict[str, Pipeline] = {
                "Regresión Logística": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
                            ),
                        ),
                    ]
                ),
                "Random Forest": Pipeline(
                    [
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=200,
                                max_depth=6,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        )
                    ]
                ),
            }

            if XGBOOST_AVAILABLE:
                model_pipelines["XGBoost"] = Pipeline(
                    [
                        (
                            "model",
                            xgb.XGBClassifier(
                                n_estimators=200,
                                max_depth=4,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                eval_metric="logloss",
                                verbosity=0,
                            ),
                        )
                    ]
                )

            rows: List[Dict[str, str]] = []
            self.roc_data = []

            for name, pipe in model_pipelines.items():
                pipe.fit(X_train, y_train)
                proba: np.ndarray = pipe.predict_proba(X_test)[:, 1]

                auc = float(roc_auc_score(y_test, proba))
                fpr, tpr, _ = roc_curve(y_test, proba)
                self.roc_data.append(
                    {"name": name, "fpr": fpr, "tpr": tpr, "auc": auc}
                )

                rows.append(
                    {
                        "Modelo": name,
                        "AUC-ROC": f"{auc:.4f}",
                        "Gini": f"{self._compute_gini(y_test, proba):.4f}",
                        "KS": f"{self._compute_ks(y_test, proba):.4f}",
                        "Odds Ratio": self._compute_odds_ratio(pipe),
                    }
                )

            return pd.DataFrame(rows), self._build_roc_figure()

        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"Error inesperado durante el entrenamiento: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Clase DebtAdvisor
# ──────────────────────────────────────────────────────────────────────────────


class DebtAdvisor:
    """Asistente inteligente de cobranzas basado en Google Gemini.

    Conduce una conversación guiada para recopilar los cuatro datos financieros
    clave del usuario (monto de deuda, días de mora, gastos mensuales e ingresos)
    y genera un plan de pagos personalizado, empático y financieramente viable.

    Attributes:
        _client: Cliente de Google Gemini configurado con la API Key.
        SYSTEM_PROMPT: Instrucciones del sistema que definen el comportamiento
                       del asesor de cobranzas.

    Example:
        >>> advisor = DebtAdvisor()
        >>> history, session, _ = advisor.chat("Mi deuda es $5000", [], None)
    """

    SYSTEM_PROMPT: str = (
        "Eres un asesor financiero especializado en cobranzas y recuperación de deuda, "
        "con un enfoque empático, profesional y orientado a soluciones sostenibles.\n\n"
        "## FLUJO OBLIGATORIO DE RECOPILACIÓN DE DATOS\n"
        "Antes de generar cualquier plan, DEBES confirmar estos 4 datos en orden:\n"
        "1. Monto total de la deuda (en la moneda del usuario)\n"
        "2. Días de mora (tiempo transcurrido sin realizar pagos)\n"
        "3. Gastos familiares mensuales totales (egresos fijos + variables)\n"
        "4. Ingresos mensuales netos (después de impuestos)\n\n"
        "- Solicita un dato a la vez de forma cordial y clara.\n"
        "- Si el usuario proporciona varios datos a la vez, extráelos todos y confirma.\n"
        "- Valida que los valores sean numéricos y positivos antes de aceptarlos.\n"
        "- NO generes el plan hasta tener los 4 datos explícitamente confirmados.\n\n"
        "## ESTRUCTURA DEL PLAN DE PAGOS\n"
        "Una vez confirmados los 4 datos, genera un plan con estas secciones exactas:\n\n"
        "### 📊 DIAGNÓSTICO FINANCIERO\n"
        "- Capacidad de pago real (ingresos − gastos)\n"
        "- Relación deuda/ingresos expresada en porcentaje\n"
        "- Clasificación del nivel de riesgo: Bajo / Medio / Alto / Crítico\n\n"
        "### 📅 PLAN DE PAGOS PROPUESTO\n"
        "- Cuota mensual sugerida (máx. 30% del ingreso disponible)\n"
        "- Plazo estimado de liquidación total de la deuda\n"
        "- Estrategia de negociación específica si mora > 60 días\n\n"
        "### 💡 RECOMENDACIONES DE AHORRO\n"
        "- Exactamente 4 acciones concretas y accionables para liberar flujo de caja\n"
        "- Prioridades de pago si el usuario menciona múltiples obligaciones\n\n"
        "### ✅ RESUMEN EJECUTIVO\n"
        "- Tabla markdown con: Deuda total | Cuota mensual | Plazo | Ahorro estimado\n\n"
        "## TONO Y ESTILO\n"
        "- Empático y alentador, nunca intimidante ni alarmista.\n"
        "- Usa cifras concretas, nunca generalidades vagas.\n"
        "- Si la situación es financieramente crítica, expresa urgencia con comprensión.\n"
        "- Usa markdown con emojis moderados para mejorar la legibilidad.\n"
    )

    def __init__(self) -> None:
        """Inicializa DebtAdvisor y configura el cliente Gemini si está disponible."""
        self._client: Optional[Any] = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self._client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as exc:
                print(f"[DebtAdvisor] Error inicializando Gemini: {exc}")

    @staticmethod
    def initial_history() -> List[Dict[str, str]]:
        """Retorna el historial inicial del chat con el mensaje de bienvenida.

        Returns:
            Lista con un único mensaje de apertura del asistente en formato
            messages de Gradio (dicts con 'role' y 'content').
        """
        welcome = (
            "👋 **Bienvenido al Asesor Inteligente de Cobranzas**\n\n"
            "Mi objetivo es ayudarte a diseñar un **plan de pagos personalizado** "
            "que se ajuste a tu situación real, de forma empática y sin juicios.\n\n"
            "Para elaborar tu plan necesitaré cuatro datos clave:\n\n"
            "| # | Información requerida |\n"
            "|---|----------------------|\n"
            "| 1 | 💰 Monto total de tu deuda |\n"
            "| 2 | 📅 Días de mora (tiempo sin pagar) |\n"
            "| 3 | 🏠 Gastos familiares mensuales totales |\n"
            "| 4 | 💼 Ingresos mensuales netos |\n\n"
            "Toda la información compartida es tratada con total **confidencialidad** "
            "— esta sesión no es almacenada.\n\n"
            "¿Por dónde comenzamos? Cuéntame: **¿cuál es el monto total de tu deuda?** 💬"
        )
        return [{"role": "assistant", "content": welcome}]

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        session: Optional[Any],
    ) -> Tuple[List[Dict[str, str]], Optional[Any], str]:
        """Procesa el mensaje del usuario y obtiene la respuesta del asistente.

        Mantiene la sesión de chat de Gemini entre turnos para preservar
        el contexto de la conversación. Si la API no está disponible, devuelve
        mensajes de error descriptivos en lugar de lanzar excepciones.

        Args:
            message: Texto ingresado por el usuario en el campo de texto.
            history: Historial de conversación en formato messages de Gradio.
            session: Objeto de sesión de chat de Gemini, o None si es el primer turno.

        Returns:
            Tupla con:
                - Historial actualizado con el nuevo intercambio usuario/asistente.
                - Sesión de chat de Gemini actualizada (o la misma si falló).
                - String vacío para limpiar el campo de entrada tras el envío.
        """
        if not message or not message.strip():
            return history, session, ""

        if not GEMINI_AVAILABLE:
            gr.Warning("El paquete 'google-generativeai' no está instalado.")
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": (
                            "⚠️ El paquete requerido no está instalado.\n"
                            "Ejecuta: `pip install google-generativeai`"
                        ),
                    },
                ],
                session,
                "",
            )

        if not GEMINI_API_KEY:
            gr.Warning(
                "Configure la variable de entorno GEMINI_API_KEY para activar el asistente."
            )
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": (
                            "⚠️ **API Key no configurada.**\n\n"
                            "Defina la variable de entorno `GEMINI_API_KEY` "
                            "y reinicie la aplicación para habilitar este módulo."
                        ),
                    },
                ],
                session,
                "",
            )

        if self._client is None:
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": (
                            "⚠️ El modelo de IA no pudo iniciarse. "
                            "Verifique que su API Key sea válida y tenga permisos activos."
                        ),
                    },
                ],
                session,
                "",
            )

        try:
            if session is None:
                session = self._client.chats.create(
                    model=GEMINI_MODEL_NAME,
                    config=GenerateContentConfig(system_instruction=self.SYSTEM_PROMPT),
                )
            response = session.send_message(message)
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response.text},
                ],
                session,
                "",
            )
        except Exception as exc:
            gr.Warning(f"Error al comunicarse con Gemini API: {exc}")
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": (
                            f"⚠️ Error de comunicación con el servicio de IA: {exc}\n\n"
                            "Por favor, intenta de nuevo en unos momentos."
                        ),
                    },
                ],
                session,
                "",
            )

    def reset(self) -> Tuple[List[Dict[str, str]], None]:
        """Reinicia el chat eliminando el historial y cerrando la sesión activa.

        Returns:
            Tupla con el historial de bienvenida inicial y None como nueva sesión.
        """
        return self.initial_history(), None


# ──────────────────────────────────────────────────────────────────────────────
# Conclusiones IA — Módulo 1
# ──────────────────────────────────────────────────────────────────────────────


def generate_ai_conclusions(metrics_df: pd.DataFrame) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "⚠️ Configure `GEMINI_API_KEY` para obtener conclusiones generadas por IA."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        table_md = metrics_df.to_markdown(index=False)
        prompt = (
            "Eres un experto en modelado de riesgo crediticio y scorecard bancario.\n"
            "Analiza los siguientes resultados de entrenamiento de modelos de clasificación crediticia:\n\n"
            f"{table_md}\n\n"
            "Genera un análisis profesional estructurado con exactamente estas tres secciones en markdown:\n\n"
            "### 📊 Análisis por Modelo\n"
            "Interpreta los resultados de cada modelo individualmente: capacidad discriminante, "
            "fortalezas y debilidades según sus métricas AUC-ROC, Gini y KS.\n\n"
            "### 📈 Comparativa de Métricas de Desempeño\n"
            "Compara los modelos entre sí. Referencia los rangos aceptables en la industria financiera "
            "(Gini > 0.3 aceptable, > 0.5 bueno; KS > 0.2 aceptable, > 0.4 bueno). "
            "Señala brechas significativas entre modelos.\n\n"
            "### ✅ Modelo Recomendado para Implementación\n"
            "Indica cuál modelo implementar con justificación técnica basada en las métricas obtenidas. "
            "Considera también interpretabilidad regulatoria (Basilea, IFRS 9) y costo computacional. "
            "Si el Odds Ratio está disponible, úsalo en el argumento.\n\n"
            "Sé conciso, usa cifras concretas del análisis y responde en español."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        return f"⚠️ Error al generar conclusiones: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Instancias globales
# ──────────────────────────────────────────────────────────────────────────────

trainer = ModelTrainer()
advisor = DebtAdvisor()

# ──────────────────────────────────────────────────────────────────────────────
# Estilos CSS
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.header-block {
    background: linear-gradient(135deg, #0f2444 0%, #1d4ed8 100%);
    padding: 30px 40px;
    border-radius: 14px;
    margin-bottom: 6px;
    box-shadow: 0 4px 20px rgba(29, 78, 216, 0.25);
}
.header-block h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: #ffffff !important;
    letter-spacing: -0.02em;
}
.header-block p {
    font-size: 0.93rem;
    opacity: 0.80;
    margin: 0;
    color: #e2e8f0 !important;
}
.footnote {
    font-size: 0.78rem;
    color: #6b7280;
    font-style: italic;
    margin-top: 4px;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Interfaz Gradio con gr.Blocks
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Plataforma de Análisis Crediticio") as demo:

    # ── Encabezado ─────────────────────────────────────────────────────────────
    gr.HTML(
        """
        <div class="header-block">
            <h1>🏦 Plataforma Integral de Análisis Crediticio</h1>
            <p>Modelado Estadístico de Riesgo Crediticio &nbsp;·&nbsp;
               Asesoría Inteligente de Cobranzas con IA</p>
        </div>
        """
    )

    with gr.Tabs():

        # ── TAB 1: Modelado Estadístico de Riesgo ──────────────────────────────
        with gr.Tab("📊 Modelado de Riesgo Crediticio"):

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### 📁 Carga de Datos")
                    file_input = gr.File(
                        label="Archivo de datos (.csv o .txt)",
                        file_types=[".csv", ".txt"],
                    )
                    upload_status = gr.Textbox(
                        label="Estado de carga",
                        interactive=False,
                        placeholder="Esperando carga de archivo…",
                        lines=1,
                    )

                with gr.Column(scale=2):
                    gr.Markdown("#### 👁️ Vista Previa del Dataset")
                    data_preview = gr.Dataframe(
                        label="Primeras 10 filas",
                        wrap=True,
                        max_height=260,
                    )

            gr.Markdown("---")
            gr.Markdown("#### ⚙️ Configuración del Modelo")

            with gr.Row():
                target_dd = gr.Dropdown(
                    label="🎯 Variable Objetivo (Target)",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="Variable binaria a predecir — debe tener exactamente 2 clases (0 / 1)",
                    scale=1,
                )
                features_dd = gr.Dropdown(
                    label="📐 Variables Explicativas (Features)",
                    choices=[],
                    multiselect=True,
                    interactive=True,
                    info="Selecciona uno o más predictores para incluir en los modelos",
                    scale=2,
                )

            train_btn = gr.Button(
                "🚀  Entrenar y Evaluar Modelos",
                variant="primary",
                size="lg",
            )

            gr.Markdown("---")

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("#### 📋 Métricas Comparativas")
                    metrics_tbl = gr.Dataframe(
                        label="Indicadores de Discriminación por Modelo",
                        wrap=False,
                        max_height=220,
                    )
                    gr.HTML(
                        '<p class="footnote">'
                        "Gini = 2·AUC−1 &nbsp;·&nbsp; KS = máx(TPR−FPR) &nbsp;·&nbsp; "
                        "Odds Ratio aplica únicamente para Regresión Logística"
                        "</p>"
                    )

                with gr.Column():
                    gr.Markdown("#### 📉 Curva ROC Comparativa")
                    roc_plot = gr.Plot(label="Curvas ROC por Modelo")

            gr.Markdown("---")
            gr.Markdown("#### 🤖 Conclusiones Generadas por IA")
            ai_conclusions = gr.Markdown(
                value="*Entrena los modelos para obtener el análisis automático.*"
            )

        # ── TAB 2: Asistente Inteligente de Cobranzas ──────────────────────────
        with gr.Tab("💬 Asistente Inteligente de Cobranzas"):

            gr.Markdown(
                "#### 🤖 Asesor de Cobranzas &nbsp;·&nbsp; *Powered by Google Gemini*\n\n"
                "Responde las preguntas del asistente para recibir un plan de pagos "
                "personalizado basado en tu situación financiera real."
            )

            session_state = gr.State(None)

            chatbot_ui = gr.Chatbot(
                label="Sesión de Asesoría Financiera",
                value=advisor.initial_history(),
                height=490,
                avatar_images=(None, "🏦"),
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Escribe aquí tu respuesta y presiona Enter o el botón Enviar…",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button(
                    "Enviar ➤",
                    variant="primary",
                    scale=1,
                    min_width=110,
                )

            with gr.Row():
                reset_btn = gr.Button(
                    "🔄 Nueva Consulta",
                    variant="secondary",
                    size="sm",
                )
                gr.HTML(
                    '<p class="footnote" style="margin-top:8px;">'
                    "🔒 Esta conversación no es almacenada. Tus datos son confidenciales."
                    "</p>"
                )

    # ── Manejadores de eventos ──────────────────────────────────────────────────

    def _on_file_upload(
        file: Any,
    ) -> Tuple[Optional[pd.DataFrame], Any, Any, str]:
        """Gestiona la carga del archivo, actualiza la vista previa y los dropdowns.

        Args:
            file: Archivo subido por el usuario mediante gr.File.

        Returns:
            Tupla con preview DataFrame, update para target_dd,
            update para features_dd y mensaje de estado.
        """
        if file is None:
            return (
                None,
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
                "",
            )
        preview, cols, msg = trainer.load_data(file)
        return (
            preview,
            gr.update(choices=cols, value=None),
            gr.update(choices=cols, value=[]),
            msg,
        )

    file_input.change(
        fn=_on_file_upload,
        inputs=[file_input],
        outputs=[data_preview, target_dd, features_dd, upload_status],
    )

    def _on_train(
        target: Optional[str],
        features: Optional[List[str]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[Any], str]:
        if not target:
            gr.Warning("Seleccione una variable objetivo antes de entrenar.")
            return None, None, ""
        if not features:
            gr.Warning("Seleccione al menos una variable explicativa.")
            return None, None, ""
        metrics_df, fig = trainer.train_and_evaluate(target, list(features))
        conclusions = generate_ai_conclusions(metrics_df)
        return metrics_df, fig, conclusions

    train_btn.click(
        fn=_on_train,
        inputs=[target_dd, features_dd],
        outputs=[metrics_tbl, roc_plot, ai_conclusions],
    )

    def _on_send(
        msg: str,
        history: List[Dict[str, str]],
        session: Optional[Any],
    ) -> Tuple[List[Dict[str, str]], Optional[Any], str]:
        """Envía el mensaje del usuario al asesor y actualiza el chat.

        Args:
            msg: Texto del campo de entrada del usuario.
            history: Historial actual del chatbot en formato messages.
            session: Sesión activa de Gemini (o None para el primer turno).

        Returns:
            Tupla con historial actualizado, sesión actualizada y string vacío.
        """
        return advisor.chat(msg, history, session)

    send_btn.click(
        fn=_on_send,
        inputs=[msg_box, chatbot_ui, session_state],
        outputs=[chatbot_ui, session_state, msg_box],
    )
    msg_box.submit(
        fn=_on_send,
        inputs=[msg_box, chatbot_ui, session_state],
        outputs=[chatbot_ui, session_state, msg_box],
    )

    def _on_reset() -> Tuple[List[Dict[str, str]], None]:
        """Reinicia la sesión del chatbot al estado inicial de bienvenida.

        Returns:
            Tupla con historial de bienvenida y None como nueva sesión.
        """
        return advisor.reset()

    reset_btn.click(
        fn=_on_reset,
        outputs=[chatbot_ui, session_state],
    )

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=_CSS,
    )
