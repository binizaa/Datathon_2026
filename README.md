# Hey Banco Datathon 2026 — Algoritmos

Este repositorio contiene los algoritmos desarrollados para el reto de segmentación de clientes y análisis de conversaciones de Havi, el asistente virtual de Hey Banco.

---

## Archivos

| Archivo | Tipo | Descripción |
|---|---|---|
| `02_havi_embeddings_analysis.py` | Script Python | Pipeline NLP de tópicos en conversaciones |
| `clusters_pipeline_v3_FINAL.ipynb` | Notebook | Segmentación de clientes (pipeline final) |
| `notificaciones.ipynb` | Notebook | TDA Mapper + Motor de avisos financieros |

---

## 1. `02_havi_embeddings_analysis.py` — Análisis NLP de Havi con Embeddings

Descubre tópicos no supervisados en las conversaciones del chatbot usando embeddings semánticos.

### Pipeline

| Paso | Técnica | Detalle |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384 dimensiones, soporte español mexicano |
| Reducción | UMAP 10D | Para clustering (preserva estructura) |
| Reducción | UMAP 2D | Para visualización |
| Clustering | HDBSCAN | `min_cluster_size=100`, selección EOM |
| Naming | Frecuencia de palabras clave | Top 3 palabras por clúster |

### Outputs generados

- `havi_embeddings.npy` — vectores 384D de todos los mensajes
- `havi_user_topics.csv` — tópico dominante + distribución proporcional por usuario
- `havi_topics_umap.png` — mapa 2D de tópicos
- `havi_topics_barplot.png` — distribución de tópicos por volumen

### Features que produce (para el modelo de segmentación)

```
topic_dominante          # tópico más frecuente del usuario
n_topics_distintos       # diversidad de consultas
n_msgs_con_topico        # cobertura de mensajes asignados
pct_topic_<nombre>       # proporción de cada tópico (one-hot suave)
```

### Dependencias

```bash
pip install sentence-transformers umap-learn hdbscan
```

---

## 2. `clusters_pipeline_v3_FINAL.ipynb` — Pipeline de Segmentación V3

Pipeline final de clustering de clientes bancarios. Se probaron **39 configuraciones** y se eligió la de mayor utilidad de negocio.

### Feature Engineering

7 features seleccionadas (RFM + comportamiento):

| Feature | Descripción |
|---|---|
| `recency_dias` | Días desde última transacción |
| `frequency` | Total de transacciones |
| `monetary` | Suma de montos |
| `dias_activos` | Días distintos con actividad |
| `monto_promedio_compra` | Ticket promedio |
| `n_categorias` | Diversidad de categorías |
| `pct_digital` | Proporción de transacciones digitales |

### Preprocessing

- Winsorización al 2% en colas largas
- `QuantileTransformer` (más robusto que PowerTransformer para datos bancarios)

### Algoritmo elegido

**UMAP 3D + KMeans k=5**

| Métrica | Valor |
|---|---|
| Silhouette Score | 0.591 |
| Clusters | 5 |

### Segmentos de clientes

| ID | Segmento | Descripción |
|---|---|---|
| 0 | **Mainstream Joven** | Activo digital, sin productos avanzados. Candidato a inversión. |
| 1 | **Premium Inversionista** | Multiproducto, 57% con inversión. Estrategia de wallet share. |
| 2 | **Cliente Distraído** | +45 días sin login, frecuencia baja. Requiere reactivación. |
| 3 | **Empresario** | Perfil de negocio, transacciones de mayor monto. |
| 4 | *(5to segmento)* | Definido por el grid search. |

### Por qué UMAP sobre PCA

UMAP preserva estructura no-lineal y produce clusters más separables en datos bancarios con distribuciones de cola larga. PCA asume linealidad y pierde esa estructura.

### Dependencias

```bash
pip install umap-learn scikit-learn pandas numpy matplotlib
```

---

## 3. `notificaciones.ipynb` — TDA Mapper + Motor de Avisos

Notebook con dos algoritmos independientes.

### 3a. TDA Mapper — Análisis Topológico de Clientes

Usa **Topological Data Analysis (TDA)** con la librería `kmapper` para construir un grafo de clientes y detectar "islas" o grupos topológicamente distintos.

| Componente | Técnica |
|---|---|
| Lens (filtro) | PCA / engagement score |
| Clustering interno | DBSCAN |
| Grafo | NetworkX |
| Visualización | TSNE 2D |

El grafo revela subestructuras que los métodos de clustering plano (KMeans, HDBSCAN) no capturan, como clientes de transición entre segmentos.

### 3b. Avisos Engine — Motor de Alertas Financieras

Pipeline de 5 pasos sobre `hey_transacciones.csv` que genera `avisos_output.json` con alertas personalizadas por usuario.

| Paso | Alerta | Lógica |
|---|---|---|
| 1 | Recurrencias detectadas | Mismo comercio + monto similar + 3 o más veces |
| 2 | Suscripción que subió de precio | Cargo recurrente con incremento mes a mes > $10 |
| 3 | Forecast de cashflow 30 días | Ingresos vs gastos promedio histórico |
| 4 | *(pasos adicionales)* | Configurables en el pipeline |

Output: `avisos_output.json` con alertas y reportes listos para consumir desde la app.

### Dependencias

```bash
pip install kmapper networkx scikit-learn pandas numpy matplotlib seaborn
```

---

## Flujo general entre algoritmos

```
chat.csv
    └──► 02_havi_embeddings_analysis.py
              └──► havi_user_topics.csv
                        └──► clusters_pipeline_v3_FINAL.ipynb
                                  └──► Segmentos de clientes (5 clusters)

hey_transacciones.csv
    └──► notificaciones.ipynb (Avisos Engine)
              └──► avisos_output.json

hey_clientes.csv + hey_productos.csv
    └──► clusters_pipeline_v3_FINAL.ipynb (perfil enriquecido)
    └──► notificaciones.ipynb (TDA Mapper)
```
