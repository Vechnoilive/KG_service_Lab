# -*- coding: utf-8 -*-
import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("[HYBRID] Инициализация...", file=sys.stderr, flush=True)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
os.environ['PYDANTIC_V1_COMPATIBILITY_MODE'] = '1'

try:
    import spacy
    print("[HYBRID] ✓ spacy импортирован", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"[HYBRID] ✗ Ошибка импорта spacy: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
except Exception as e:
    print(f"[HYBRID] ✗ Ошибка при загрузке spacy: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

import json
import csv
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import math

nlp = None
try:
    print("[HYBRID] Попытка загрузить ru_core_news_lg...", file=sys.stderr, flush=True)
    nlp = spacy.load("ru_core_news_lg")
    print("[HYBRID] ✓ Модель ru_core_news_lg загружена", file=sys.stderr, flush=True)
except OSError as e:
    print(f"[HYBRID] Ошибка загрузки ru_core_news_lg: {e}", file=sys.stderr, flush=True)
    try:
        print("[HYBRID] Попытка загрузить ru_core_news_sm...", file=sys.stderr, flush=True)
        nlp = spacy.load("ru_core_news_sm")
        print("[HYBRID] ✓ Модель ru_core_news_sm загружена", file=sys.stderr, flush=True)
    except OSError as e2:
        print(f"[HYBRID] ✗ КРИТИЧЕСКАЯ ОШИБКА: {e2}", file=sys.stderr, flush=True)
        print("[HYBRID] Установите модель: python -m spacy download ru_core_news_sm", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[HYBRID] ✗ Неожиданная ошибка: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if nlp is None:
    print("[HYBRID] ✗ КРИТИЧЕСКАЯ ОШИБКА: spaCy модель не загружена", file=sys.stderr, flush=True)
    sys.exit(1)

print("[HYBRID] ✓ Все зависимости загружены", file=sys.stderr, flush=True)

SEMANTIC_SIMILARITY_THRESHOLD = 0.30
ENTITY_SIMILARITY_THRESHOLD = 0.40
ENTITY_WEIGHT_MULTIPLIER = 1.5
COOCCURRENCE_WINDOW_SIZE = 8
COOCCURRENCE_MAX_WEIGHT = 4.0
SYNTAX_WEIGHT_MULTIPLIER = 0.8
SYNTAX_MAX_WEIGHT = 3.5
TFIDF_TOP_WORDS = 25
TFIDF_MULTIPLIER = 100
TFIDF_MAX_WEIGHT = 3.5
FINAL_MAX_WEIGHT = 12.0
MAX_RELATIONS_TO_SHOW = 3
MIN_TOKEN_LENGTH = 2
MIN_ENTITY_LENGTH = 3
TOP_NODES_COUNT = 25
NODE_SIZE = 35
NODE_COLOR_HYBRID = "#27ae60"

RUSSIAN_STOPWORDS = {
    'а', 'и', 'или', 'но', 'в', 'на', 'с', 'из', 'к', 'для', 'по', 'у', 'не', 'да',
    'это', 'то', 'что', 'как', 'быть', 'иметь', 'есть', 'он', 'она', 'оно', 'они',
    'мы', 'ты', 'я', 'вы', 'его', 'её', 'их', 'мне', 'тебе', 'ему', 'ей', 'нам',
    'вам', 'им', 'самый', 'весь', 'целый', 'один', 'первый', 'второй', 'третий',
    'со', 'над', 'под', 'между', 'около', 'перед', 'после', 'через', 'при'
}

def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity"""
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def extract_keywords_with_vectors(text: str) -> Dict[str, List[float]]:
    """Extract keywords with vectors"""
    if nlp is None:
        raise RuntimeError("spaCy модель не загружена")
    
    doc = nlp(text)
    keyword_vectors = {}
    
    for token in doc:
        if token.lemma_ in RUSSIAN_STOPWORDS or len(token.lemma_) < MIN_TOKEN_LENGTH:
            continue
        if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'):
            if token.has_vector:
                keyword_vectors[token.lemma_] = token.vector.tolist()
    
    print(f"[HYBRID] Извлечено {len(keyword_vectors)} ключевых слов", file=sys.stderr, flush=True)
    return keyword_vectors

def extract_named_entities(text: str) -> List[str]:
    """Extract named entities"""
    if nlp is None:
        return []
    
    doc = nlp(text)
    entities = set()
    
    for ent in doc.ents:
        lemma = ent.text.lower()
        if len(lemma) >= MIN_ENTITY_LENGTH:
            entities.add(lemma)
    
    print(f"[HYBRID] Найдено {len(entities)} сущностей", file=sys.stderr, flush=True)
    return list(entities)

def extract_semantic_edges(text: str, keyword_vectors: Dict) -> List[Tuple[str, str, str, float]]:
    """Extract semantic edges"""
    keywords = list(keyword_vectors.keys())
    edges = []
    
    for i, word1 in enumerate(keywords):
        for word2 in keywords[i+1:]:
            similarity = cosine_similarity(keyword_vectors[word1], keyword_vectors[word2])
            if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                edges.append((word1, word2, f"semantic:{similarity:.2f}", similarity))
    
    return edges

def extract_cooccurrence_edges(text: str) -> List[Tuple[str, str, str, float]]:
    """Extract co-occurrence edges"""
    if nlp is None:
        return []
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.lemma_ not in RUSSIAN_STOPWORDS and len(token.lemma_) >= MIN_TOKEN_LENGTH:
            if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'):
                tokens.append(token.lemma_)
    
    cooccurrence_weights = defaultdict(int)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + COOCCURRENCE_WINDOW_SIZE, len(tokens))):
            word1, word2 = tokens[i], tokens[j]
            if word1 != word2:
                key = tuple(sorted([word1, word2]))
                cooccurrence_weights[key] += 1
    
    edges = []
    for (word1, word2), count in cooccurrence_weights.items():
        if count >= 1:
            weight = min(float(count), COOCCURRENCE_MAX_WEIGHT)
            edges.append((word1, word2, f"cooccurrence:{count}", weight))
    
    return edges

def extract_syntactic_edges(text: str) -> List[Tuple[str, str, str, float]]:
    """Extract syntactic edges"""
    if nlp is None:
        return []
    
    doc = nlp(text)
    syntactic_weights = defaultdict(lambda: {'relation': '', 'count': 0})
    
    for token in doc:
        if token.head != token:
            head_lemma = token.head.lemma_
            child_lemma = token.lemma_
            if (head_lemma not in RUSSIAN_STOPWORDS and child_lemma not in RUSSIAN_STOPWORDS and
                len(head_lemma) >= MIN_TOKEN_LENGTH and len(child_lemma) >= MIN_TOKEN_LENGTH):
                key = tuple(sorted([head_lemma, child_lemma]))
                syntactic_weights[key]['relation'] = token.dep_
                syntactic_weights[key]['count'] += 1
    
    edges = []
    for (word1, word2), data in syntactic_weights.items():
        weight = min(float(data['count']) * SYNTAX_WEIGHT_MULTIPLIER, SYNTAX_MAX_WEIGHT)
        edges.append((word1, word2, f"syntax:{data['relation']}", weight))
    
    return edges

def extract_entity_edges(entities: List[str], keyword_vectors: Dict) -> List[Tuple[str, str, str, float]]:
    """Extract entity edges"""
    edges = []
    
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i+1:]:
            if ent1 in keyword_vectors and ent2 in keyword_vectors:
                similarity = cosine_similarity(keyword_vectors[ent1], keyword_vectors[ent2])
                if similarity >= ENTITY_SIMILARITY_THRESHOLD:
                    weight = similarity * ENTITY_WEIGHT_MULTIPLIER
                    edges.append((ent1, ent2, f"entity:{similarity:.2f}", weight))
    
    return edges

def create_html_visualization(edges: List[Tuple[str, str, str, float]], nodes: List[str], html_path: str) -> bool:
    """Creates HTML visualization of the graph"""
    print(f"[HYBRID] Создаю HTML визуализацию: {html_path}", file=sys.stderr, flush=True)
    
    html_dir = os.path.dirname(html_path)
    if html_dir and not os.path.exists(html_dir):
        try:
            os.makedirs(html_dir, exist_ok=True)
        except Exception as e:
            print(f"[HYBRID] ⚠️ Ошибка создания директории: {e}", file=sys.stderr, flush=True)
    
    try:
        nodes_data = []
        for i, node in enumerate(nodes):
            nodes_data.append({
                "id": i,
                "label": node,
                "title": node,
                "size": 30,
                "color": NODE_COLOR_HYBRID
            })
        
        edges_data = []
        node_map = {n: i for i, n in enumerate(nodes)}
        
        for source, target, relation, weight in edges:
            if source in node_map and target in node_map:
                edges_data.append({
                    "from": node_map[source],
                    "to": node_map[target],
                    "label": relation[:15],
                    "weight": float(weight),
                    "title": f"{relation} (вес: {weight:.2f})",
                    "color": {"color": NODE_COLOR_HYBRID},
                    "arrows": "to"
                })
        
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Гибридный граф</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        #network {{ width: 100%; height: 100vh; border: 1px solid lightgray; }}
        .info {{ position: absolute; top: 10px; left: 10px; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); z-index: 10; }}
    </style>
</head>
<body>
    <div class="info">
        <h3>Гибридный граф</h3>
        <p>Узлов: {len(nodes)}</p>
        <p>Рёбер: {len(edges)}</p>
    </div>
    <div id="network"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            physics: {{ enabled: true, stabilization: {{ iterations: 200 }} }},
            interaction: {{ navigationButtons: true, keyboard: true }},
            layout: {{ randomSeed: 42 }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[HYBRID] ✓ HTML сохранён", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[HYBRID] ✗ Ошибка HTML: {e}", file=sys.stderr, flush=True)
        return False

def build_hybrid_graph(text: str):
    """Build hybrid graph combining all analysis types"""
    print("[HYBRID] Извлекаю ключевые слова...", file=sys.stderr, flush=True)
    keyword_vectors = extract_keywords_with_vectors(text)
    
    print("[HYBRID] Извлекаю сущности...", file=sys.stderr, flush=True)
    entities = extract_named_entities(text)
    
    print("[HYBRID] Извлекаю семантические связи...", file=sys.stderr, flush=True)
    semantic_edges = extract_semantic_edges(text, keyword_vectors)
    
    print("[HYBRID] Извлекаю синтаксические связи...", file=sys.stderr, flush=True)
    syntactic_edges = extract_syntactic_edges(text)
    
    print("[HYBRID] Извлекаю связи сопровождения...", file=sys.stderr, flush=True)
    cooccurrence_edges = extract_cooccurrence_edges(text)
    
    print("[HYBRID] Извлекаю связи сущностей...", file=sys.stderr, flush=True)
    entity_edges = extract_entity_edges(entities, keyword_vectors)
    
    all_edges = semantic_edges + syntactic_edges + cooccurrence_edges + entity_edges
    
    nodes_set = set()
    for source, target, _, _ in all_edges:
        nodes_set.add(source)
        nodes_set.add(target)
    
    nodes = sorted(list(nodes_set))
    
    print(f"[HYBRID] Всего узлов: {len(nodes)}", file=sys.stderr, flush=True)
    print(f"[HYBRID] Всего рёбер: {len(all_edges)}", file=sys.stderr, flush=True)
    
    return all_edges, nodes

def main():
    """Main function - ✅ ИСПРАВЛЕНИЕ 6: Читаем из файла, а не из stdin"""
    try:
        if len(sys.argv) < 2:
            print("[HYBRID] ✗ ОШИБКА: укажите путь к JSON файлу", file=sys.stderr, flush=True)
            sys.exit(1)
        
        json_file = sys.argv[1]
        print(f"[HYBRID] Читаю JSON из файла: {json_file}", file=sys.stderr, flush=True)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            json_input = json.load(f)
        
        print(f"[HYBRID] ✓ JSON загружен успешно", file=sys.stderr, flush=True)
        
    except FileNotFoundError:
        print(f"[HYBRID] ✗ ОШИБКА: файл не найден: {json_file}", file=sys.stderr, flush=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[HYBRID] ✗ ОШИБКА парсинга JSON: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[HYBRID] ✗ ОШИБКА чтения: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    text = json_input.get('text', '')
    output_csv = json_input.get('output_csv', './output.csv')
    output_html = json_input.get('output_html', './output.html')
    
    if not text:
        print("[HYBRID] ✗ ОШИБКА: текст пуст", file=sys.stderr, flush=True)
        sys.exit(1)
    
    print(f"[HYBRID] Обрабатываю текст ({len(text)} символов)...", file=sys.stderr, flush=True)
    
    edges, nodes = build_hybrid_graph(text)
    
    if edges:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'relation', 'weight'])
            for source, target, relation, weight in edges:
                writer.writerow([source, target, relation, f"{weight:.2f}"])
        print(f"[HYBRID] ✓ CSV сохранён", file=sys.stderr, flush=True)
    
    create_html_visualization(edges, nodes, output_html)
    
    print("[HYBRID] ✓ Готово!", file=sys.stderr, flush=True)
    print(json.dumps({
        "status": "success",
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "csv_file": output_csv,
        "html_file": output_html
    }))

if __name__ == "__main__":
    main()