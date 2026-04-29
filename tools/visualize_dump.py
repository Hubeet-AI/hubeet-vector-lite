import struct
import argparse
import sys
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

def read_hvl_dump(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic != b'HVL2':
            raise ValueError("Invalid magic header")
        
        dim, count = struct.unpack('<QQ', f.read(16))
        max_level, entry_id = struct.unpack('<Ii', f.read(8))
        
        print(f"Cargando {count} fragmentos (chunks) de dimensión {dim}...")
        
        vectors = []
        ids = []
        
        for i in range(count):
            level, id_len = struct.unpack('<II', f.read(8))
            vid = f.read(id_len).decode('utf-8', errors='replace').strip('\x00')
            data = struct.unpack(f'<{dim}f', f.read(4 * dim))
            vectors.append(data)
            ids.append(vid)
            
        return ids, vectors

def parse_obsidian_id(raw_id):
    """Extrae el nombre del archivo y el chunk de la ruta bruta."""
    doc_name = raw_id
    chunk_info = ""
    
    if "#" in raw_id:
        parts = raw_id.split("#")
        doc_name = parts[0]
        chunk_info = parts[1] if len(parts) > 1 else ""
        
    file_only = doc_name.split("/")[-1] 
    
    return doc_name, file_only, chunk_info

def main():
    parser = argparse.ArgumentParser(description="Visualizador de Mapa de Conocimiento Obsidian")
    parser.add_argument("dump_file", help="Ruta al archivo .hvl de Obsidian")
    args = parser.parse_args()

    # 1. Cargar datos
    raw_ids, vectors = read_hvl_dump(args.dump_file)
    if not vectors:
        print("¡No se encontraron vectores!")
        sys.exit(0)
        
    X = np.array(vectors)
    n_samples, dim = X.shape
    
    # 2. Procesar IDs de Obsidian
    full_paths = []
    file_names = []
    chunks = []
    
    for vid in raw_ids:
        path, file, chunk = parse_obsidian_id(vid)
        full_paths.append(path)
        file_names.append(file)
        chunks.append(chunk)

    # 3. Clustering Semántico y Nube de Palabras Temática
    print("Agrupando notas por similitud temática...")
    n_clusters = max(2, min(15, n_samples // 10)) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)

    print("Generando resumen de temas (Nube de palabras por clúster)...")
    stop_words = {'md', 'chunk', 'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 
                  'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 
                  'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'del', 'las', 'los'}
    
    cluster_summaries = {}
    
    for c in range(n_clusters):
        # Obtener rutas que pertenecen a este clúster
        rutas_en_cluster = [full_paths[i] for i in range(len(clusters)) if clusters[i] == c]
        
        palabras_cluster = []
        for ruta in rutas_en_cluster:
            # Limpiar caracteres especiales y tokenizar
            ruta_limpia = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s]', ' ', ruta.lower())
            tokens = ruta_limpia.split()
            # Filtrar
            palabras_filtradas = [w for w in tokens if w not in stop_words and not w.isdigit() and len(w) > 2]
            palabras_cluster.extend(palabras_filtradas)
            
        # Extraer las 3 palabras clave del clúster
        top_words = [word.capitalize() for word, count in Counter(palabras_cluster).most_common(3)]
        
        if top_words:
            etiqueta = f"Grupo {c}: {' | '.join(top_words)}"
        else:
            etiqueta = f"Grupo {c}: (Varios)"
            
        cluster_summaries[c] = etiqueta

    # Asignar al dataframe
    cluster_labels = [cluster_summaries[c] for c in clusters]

    # 4. Proyección a 3D
    if n_samples < 5:
        print("Usando PCA...")
        pca = PCA(n_components=min(3, n_samples))
        proj = pca.fit_transform(X)
    else:
        print("Generando mapa espacial 3D con t-SNE...")
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        proj = tsne.fit_transform(X)
        
    if proj.shape[1] == 1: proj = np.hstack([proj, np.zeros((n_samples, 2))])
    elif proj.shape[1] == 2: proj = np.hstack([proj, np.zeros((n_samples, 1))])

    # 5. Preparar DataFrame
    df = pd.DataFrame({
        'Nota': file_names,
        'Ruta_Completa': full_paths,
        'Fragmento': chunks,
        'X': proj[:, 0], 'Y': proj[:, 1], 'Z': proj[:, 2],
        'Tema_Asignado': cluster_labels
    })

    # 6. Generar Visualización
    print("Renderizando mapa interactivo...")
    fig = px.scatter_3d(
        df, x='X', y='Y', z='Z',
        color='Tema_Asignado',
        hover_name='Nota',
        hover_data={
            'X': False, 'Y': False, 'Z': False,
            'Tema_Asignado': False, 
            'Fragmento': True,
            'Ruta_Completa': True
        },
        opacity=0.85,
        title="Mapa de Conocimiento: Obsidian Hubeet Vector Lite",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(marker=dict(size=7))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend_title_text='Clústeres Temáticos'
    )
    
    out_file = "obsidian_knowledge_map.html"
    fig.write_html(out_file, full_html=True, include_plotlyjs='cdn')
    print(f"\n¡Mapa de conocimiento creado en {out_file}!")

if __name__ == '__main__':
    main()