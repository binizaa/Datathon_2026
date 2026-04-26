# ============================================================
# 🧠 Análisis NLP de Havi con Embeddings
# Hey Banco Datathon 2026
# ============================================================
# Este script usa sentence-transformers para crear embeddings
# de las conversaciones de Havi, luego HDBSCAN para descubrir
# tópicos no supervisados, y UMAP para visualizar.
#
# Corre esto en una celda de Jupyter o como script.
# ============================================================

# ---- PASO 0: Instalar dependencias ----
# Corre esto UNA VEZ en una celda separada:
# !pip install sentence-transformers umap-learn hdbscan --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Colores Hey
HEY_GREEN = '#A8E6A1'
PALETTE = ['#A8E6A1', '#7DD3C0', '#5BA9D6', '#9B7EDE', '#E091E0', '#FF8A80',
           '#FFB74D', '#81C784', '#64B5F6', '#BA68C8', '#4DB6AC', '#FF7043']

# ============================================================
# PASO 1: Cargar datos
# ============================================================
print("📂 Cargando datos de Havi...")
PATH = '/mnt/user-data/uploads/'  # Ajusta según tu entorno

havi = pd.read_csv(f'{PATH}dataset_50k_anonymized.csv', encoding='utf-8-sig')
havi['date'] = pd.to_datetime(havi['date'], errors='coerce')

# Limpiar inputs vacíos
havi['input_clean'] = havi['input'].fillna('').str.strip()
havi = havi[havi['input_clean'].str.len() > 5].copy()  # mínimo 5 chars

print(f"✅ Logs cargados: {len(havi):,}")
print(f"   Conversaciones únicas: {havi['conv_id'].nunique():,}")
print(f"   Usuarios únicos: {havi['user_id'].nunique():,}")

# ============================================================
# PASO 2: Generar embeddings con Sentence Transformers
# ============================================================
print("\n🧠 Generando embeddings (esto tarda 3-8 min en CPU)...")
print("   Modelo: paraphrase-multilingual-MiniLM-L12-v2")
print("   Maneja español mexicano, 384 dimensiones")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encodear todos los inputs
# Si es muy grande, puedes samplear: texts = havi['input_clean'].sample(20000).tolist()
texts = havi['input_clean'].tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=256,       # ajusta si tienes GPU → 512 o 1024
    normalize_embeddings=True  # normalizar para cosine similarity
)

print(f"\n✅ Embeddings generados: {embeddings.shape}")
print(f"   Dimensiones: {embeddings.shape[1]}")

# Guardar embeddings para no recalcular
import os
save_dir = os.getcwd()
np.save(os.path.join(save_dir, 'havi_embeddings.npy'), embeddings)
print(f"💾 Embeddings guardados en: {os.path.join(save_dir, 'havi_embeddings.npy')}")

# ============================================================
# PASO 3: Reducir dimensionalidad con UMAP
# ============================================================
print("\n📐 Reduciendo dimensiones con UMAP...")

import umap

# UMAP a 2D para visualización Y para clustering
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    verbose=True
)

embeddings_2d = reducer.fit_transform(embeddings)
print(f"✅ UMAP completo: {embeddings_2d.shape}")

# También 10D para clustering (más info que 2D)
reducer_cluster = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
    verbose=False
)
embeddings_10d = reducer_cluster.fit_transform(embeddings)
print(f"✅ UMAP 10D para clustering: {embeddings_10d.shape}")

# ============================================================
# PASO 4: Clustering con HDBSCAN
# ============================================================
print("\n🔬 Descubriendo tópicos con HDBSCAN...")

import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,     # mínimo 100 mensajes por tópico
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

labels = clusterer.fit_predict(embeddings_10d)
havi['topic_id'] = labels

n_topics = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"\n✅ Tópicos descubiertos: {n_topics}")
print(f"   Mensajes asignados a tópico: {(labels != -1).sum():,} ({100*(labels != -1).mean():.1f}%)")
print(f"   Mensajes sin tópico (ruido): {n_noise:,} ({100*n_noise/len(labels):.1f}%)")

# ============================================================
# PASO 5: Nombrar tópicos automáticamente
# ============================================================
print("\n🏷️ Nombrando tópicos...")

import re

stopwords_es = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para',
    'con','no','una','su','al','lo','como','más','pero','sus','le','ya','o',
    'este','sí','porque','esta','entre','cuando','muy','sin','sobre','también',
    'me','hasta','hay','donde','quien','desde','todo','nos','durante','todos',
    'uno','les','ni','contra','otros','ese','eso','ante','ellos','esto','mí',
    'antes','qué','yo','otro','otra','él','esa','estos','mucho','nada','cual',
    'poco','ella','mi','mis','te','ti','tu','tus','hola','buenas','buenos',
    'gracias','favor','puedo','puede','quiero','tengo','tiene','hacer','haber',
    'sido','ser','está','estoy','cómo','cuál','cuáles','dónde','cuánto',
    'banco','hey','cuenta','información','necesito','saber','ayuda','días'
}

def get_top_words(texts, n=8):
    """Extrae las n palabras más frecuentes de una lista de textos."""
    all_words = []
    for text in texts:
        if pd.isna(text):
            continue
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', str(text).lower())
        all_words.extend([w for w in words if w not in stopwords_es])
    return Counter(all_words).most_common(n)

topic_names = {}
topic_info = []

for topic_id in sorted(set(labels)):
    if topic_id == -1:
        continue
    
    mask = labels == topic_id
    topic_texts = havi.loc[mask, 'input_clean'].tolist()
    top_words = get_top_words(topic_texts, n=8)
    
    # Nombre automático: top 3 palabras
    name = ' | '.join([w for w, _ in top_words[:3]])
    topic_names[topic_id] = name
    
    topic_info.append({
        'topic_id': topic_id,
        'name': name,
        'n_messages': mask.sum(),
        'pct': 100 * mask.sum() / len(havi),
        'top_words': ', '.join([f"{w}({c})" for w, c in top_words]),
        'example_1': topic_texts[0][:100] if len(topic_texts) > 0 else '',
        'example_2': topic_texts[1][:100] if len(topic_texts) > 1 else '',
        'example_3': topic_texts[2][:100] if len(topic_texts) > 2 else '',
    })

topic_df = pd.DataFrame(topic_info).sort_values('n_messages', ascending=False)

# Asignar nombre al df principal
havi['topic_name'] = havi['topic_id'].map(topic_names).fillna('sin_tópico')

print("\n📊 Tópicos descubiertos (ordenados por tamaño):")
print("="*80)
for _, row in topic_df.iterrows():
    print(f"\n🏷️ Tópico {row['topic_id']}: {row['name']}")
    print(f"   Mensajes: {row['n_messages']:,} ({row['pct']:.1f}%)")
    print(f"   Palabras: {row['top_words']}")
    print(f"   Ejemplo: \"{row['example_1']}\"")

# ============================================================
# PASO 6: Visualización UMAP 2D con tópicos
# ============================================================
print("\n📊 Generando visualización...")

fig, ax = plt.subplots(figsize=(14, 10))

# Plot ruido primero (gris, pequeño)
noise_mask = labels == -1
ax.scatter(
    embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
    c='#444444', s=2, alpha=0.1, label='sin tópico'
)

# Plot cada tópico con color
for i, topic_id in enumerate(sorted(set(labels) - {-1})):
    mask = labels == topic_id
    color = PALETTE[i % len(PALETTE)]
    name = topic_names.get(topic_id, f'Topic {topic_id}')
    n = mask.sum()
    ax.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        c=color, s=8, alpha=0.5, label=f'{name} ({n:,})'
    )

ax.set_title(f'Mapa de tópicos de Havi — {n_topics} tópicos descubiertos', fontsize=14)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'havi_topics_umap.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"💾 Imagen guardada: havi_topics_umap.png")

# ============================================================
# PASO 7: Distribución de tópicos (barras)
# ============================================================
fig, ax = plt.subplots(figsize=(12, max(6, n_topics * 0.4)))

topic_counts = havi[havi['topic_id'] != -1]['topic_name'].value_counts()
colors = [PALETTE[i % len(PALETTE)] for i in range(len(topic_counts))]

ax.barh(topic_counts.index[::-1], topic_counts.values[::-1], 
        color=colors[::-1], edgecolor='black')
ax.set_title(f'Distribución de tópicos de Havi ({n_topics} tópicos)')
ax.set_xlabel('Mensajes')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'havi_topics_barplot.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PASO 8: Tópicos por usuario → features para master_full
# ============================================================
print("\n🔗 Generando features de tópicos por usuario...")

# Tópico dominante por usuario
user_topic = havi[havi['topic_id'] != -1].groupby('user_id').agg(
    topic_dominante=('topic_name', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'sin_topico'),
    n_topics_distintos=('topic_id', 'nunique'),
    n_msgs_con_topico=('topic_id', 'count'),
).reset_index()

# Distribución de tópicos por usuario (one-hot proporcional)
topic_dist = pd.crosstab(
    havi[havi['topic_id'] != -1]['user_id'],
    havi[havi['topic_id'] != -1]['topic_name'],
    normalize='index'
).reset_index()
topic_dist.columns = [f'pct_topic_{c}' if c != 'user_id' else c for c in topic_dist.columns]

user_topic = user_topic.merge(topic_dist, on='user_id', how='left')

print(f"✅ Features de tópico generadas para {len(user_topic):,} usuarios")
print(f"   Columnas: {user_topic.shape[1]}")

# Guardar
user_topic.to_csv(os.path.join(save_dir, 'havi_user_topics.csv'), index=False)
print(f"💾 Guardado: havi_user_topics.csv")

# ============================================================
# PASO 9: Cross-sell por tópico — la joya del reto
# ============================================================
print("\n🎯 Análisis de cross-sell por tópico...")
print("="*60)

# Cargar clientes y productos para cruzar
clientes = pd.read_csv(f'{PATH}hey_clientes.csv', encoding='utf-8-sig')
productos = pd.read_csv(f'{PATH}hey_productos.csv', encoding='utf-8-sig')

# Productos por usuario
for tipo in ['inversion_hey', 'tarjeta_credito_hey', 'credito_personal',
             'seguro_vida', 'seguro_compras', 'cuenta_negocios']:
    tmp = productos[productos['tipo_producto']==tipo].groupby('user_id').size()
    clientes[f'tiene_{tipo}'] = clientes['user_id'].map(tmp).fillna(0).gt(0)

# Merge con tópico dominante
cross = clientes.merge(user_topic[['user_id', 'topic_dominante']], on='user_id', how='inner')

print(f"\nUsuarios con tópico + productos: {len(cross):,}\n")

# Para cada tópico, qué productos NO tienen
for topic in cross['topic_dominante'].value_counts().head(10).index:
    topic_users = cross[cross['topic_dominante'] == topic]
    n = len(topic_users)
    print(f"\n🏷️ Tópico: '{topic}' ({n} usuarios)")
    for prod_col in ['tiene_inversion_hey', 'tiene_tarjeta_credito_hey',
                     'tiene_credito_personal', 'tiene_seguro_vida']:
        no_tiene = 100 * (1 - topic_users[prod_col].mean())
        if no_tiene > 40:  # solo mostrar si >40% no tiene
            print(f"   → {no_tiene:.0f}% NO tiene {prod_col.replace('tiene_','')}")

# ============================================================
# PASO 10: Resumen final
# ============================================================
print("\n")
print("="*60)
print("📊 RESUMEN FINAL")
print("="*60)
print(f"\n🧠 Modelo: paraphrase-multilingual-MiniLM-L12-v2")
print(f"📐 Reducción: UMAP (384D → 10D para cluster, 2D para viz)")
print(f"🔬 Clustering: HDBSCAN")
print(f"\n📈 Resultados:")
print(f"   Tópicos descubiertos: {n_topics}")
print(f"   Cobertura: {100*(labels != -1).mean():.1f}% de mensajes asignados")
print(f"   vs. baseline keywords: ~36% → ahora {100*(labels != -1).mean():.1f}%")
print(f"\n💾 Archivos generados:")
print(f"   havi_embeddings.npy    — embeddings 384D")
print(f"   havi_user_topics.csv   — features de tópico por usuario")
print(f"   havi_topics_umap.png   — mapa visual 2D")
print(f"   havi_topics_barplot.png — distribución de tópicos")
print(f"\n🔑 Para usar en el modelo de segmentación:")
print(f"   user_topic = pd.read_csv('havi_user_topics.csv')")
print(f"   master_full = master_full.merge(user_topic, on='user_id', how='left')")
