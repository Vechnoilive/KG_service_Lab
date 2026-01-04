# -*- coding: utf-8 -*-
import sys
import json
import csv
import os
from collections import Counter
from typing import List, Tuple, Dict, Set

if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("[SYNTAX] Инициализация...", file=sys.stderr, flush=True)

try:
    import spacy
    print("[SYNTAX] ✓ spacy импортирован", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"[SYNTAX] ✗ Ошибка импорта spacy: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import nltk
    print("[SYNTAX] Загружаю NLTK данные...", file=sys.stderr, flush=True)
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("[SYNTAX] Загружаю punkt...", file=sys.stderr, flush=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"[SYNTAX] NLTK ошибка: {e}", file=sys.stderr, flush=True)

nlp = None
try:
    print("[SYNTAX] Попытка загрузить ru_core_news_lg...", file=sys.stderr, flush=True)
    nlp = spacy.load("ru_core_news_lg")
    print("[SYNTAX] ✓ Модель ru_core_news_lg загружена", file=sys.stderr, flush=True)
except OSError as e:
    print(f"[SYNTAX] Ошибка загрузки ru_core_news_lg: {e}", file=sys.stderr, flush=True)
    try:
        print("[SYNTAX] Попытка загрузить ru_core_news_sm...", file=sys.stderr, flush=True)
        nlp = spacy.load("ru_core_news_sm")
        print("[SYNTAX] ✓ Модель ru_core_news_sm загружена", file=sys.stderr, flush=True)
    except OSError as e2:
        print(f"[SYNTAX] ✗ КРИТИЧЕСКАЯ ОШИБКА: {e2}", file=sys.stderr, flush=True)
        print("[SYNTAX] Установите модель: python -m spacy download ru_core_news_sm", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[SYNTAX] ✗ Неожиданная ошибка при загрузке модели: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if nlp is None:
    print("[SYNTAX] ✗ КРИТИЧЕСКАЯ ОШИБКА: spaCy модель не загружена", file=sys.stderr, flush=True)
    sys.exit(1)

print("[SYNTAX] ✓ Все зависимости загружены", file=sys.stderr, flush=True)

RUSSIAN_STOPWORDS = {
    'а', 'и', 'или', 'но', 'в', 'на', 'с', 'из', 'к', 'для', 'по', 'у', 'не', 'да',
    'это', 'то', 'что', 'как', 'быть', 'иметь', 'есть', 'он', 'она', 'оно', 'они',
    'мы', 'ты', 'я', 'вы', 'его', 'её', 'их', 'мне', 'тебе', 'ему', 'ей', 'нам',
    'вам', 'им', 'самый', 'весь', 'целый', 'один', 'первый', 'второй', 'третий',
    'со', 'над', 'под', 'между', 'около', 'перед', 'после'
}

def extract_all_keywords(text: str) -> List[str]:
    """Extracts all important keywords from text"""
    if nlp is None:
        raise RuntimeError("spaCy модель не загружена")
    
    doc = nlp(text)
    keyword_counts = Counter()
    
    for token in doc:
        if token.lemma_ in RUSSIAN_STOPWORDS or len(token.lemma_) < 3:
            continue
        if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'):
            keyword_counts[token.lemma_] += 1
    
    keywords = [kw for kw, _ in keyword_counts.most_common()]
    print(f"[SYNTAX] Извлечено {len(keywords)} ключевых слов (все леммы)", file=sys.stderr, flush=True)
    return keywords

def extract_syntax_graph(text: str) -> Tuple[List[Tuple[str, str, str, float]], List[str], List[Tuple[str, str, str, float]]]:
    """Extracts syntactic dependencies from text"""
    if nlp is None:
        raise RuntimeError("spaCy модель не загружена")
    
    print(f"[SYNTAX] Парсю текст ({len(text)} символов)...", file=sys.stderr, flush=True)
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    print(f"[SYNTAX] Обработано {num_sentences} предложений", file=sys.stderr, flush=True)
    
    print("[SYNTAX] Извлекаю ВСЕ ключевые слова...", file=sys.stderr, flush=True)
    keywords = extract_all_keywords(text)
    keywords_set = set(keywords)
    print(f"[SYNTAX] Найдено {len(keywords)} ключевых слов", file=sys.stderr, flush=True)
    
    print("[SYNTAX] Извлекаю синтаксические связи...", file=sys.stderr, flush=True)
    edges_syntax = []
    edge_weights = {}
    
    for token in doc:
        if token.head != token:
            head_lemma = token.head.lemma_
            child_lemma = token.lemma_
            
            if head_lemma in keywords_set and child_lemma in keywords_set:
                key = (head_lemma, child_lemma)
                if key not in edge_weights:
                    edge_weights[key] = {'relation': token.dep_, 'weight': 0.0}
                edge_weights[key]['weight'] += 1.0
    
    for (head, child), data in edge_weights.items():
        edges_syntax.append((head, child, f"syntax:{data['relation']}", min(data['weight'], 10.0)))
    
    print(f"[SYNTAX] Найдено {len(edges_syntax)} синтаксических связей", file=sys.stderr, flush=True)
    
    node_degrees = Counter()
    for head, child, _, _ in edges_syntax:
        node_degrees[head] += 1
        node_degrees[child] += 1
    
    all_nodes_count = len(node_degrees)
    print(f"[SYNTAX] Узлов участвующих в рёбрах: {all_nodes_count}", file=sys.stderr, flush=True)
    print(f"[SYNTAX] Всего рёбер в графе: {len(edges_syntax)}", file=sys.stderr, flush=True)
    
    top_count = min(50, len(node_degrees))
    if len(node_degrees) <= 50:
        top_count = len(node_degrees)
    
    top_nodes = set([node for node, _ in node_degrees.most_common(top_count)])
    print(f"[SYNTAX] Топ-{top_count} узлов выбраны (по степени)", file=sys.stderr, flush=True)
    
    edges_viz = []
    nodes_in_edges_viz: Set[str] = set()
    
    for head, child, relation, weight in edges_syntax:
        if head in top_nodes and child in top_nodes:
            edges_viz.append((head, child, relation, weight))
            nodes_in_edges_viz.add(head)
            nodes_in_edges_viz.add(child)
    
    final_nodes = sorted(list(nodes_in_edges_viz))
    
    print(f"[SYNTAX] ✓ Полный граф: {len(edges_syntax)} рёбер между {all_nodes_count} узлами", file=sys.stderr, flush=True)
    print(f"[SYNTAX] ✓ Визуализация: {len(edges_viz)} рёбер между {len(final_nodes)} узлами", file=sys.stderr, flush=True)
    
    return edges_viz, final_nodes, edges_syntax

def save_csv(edges_full: List[Tuple[str, str, str, float]], csv_path: str):
    """Saves edges to CSV file"""
    print(f"[SYNTAX] Сохраняю CSV (полный граф): {csv_path}", file=sys.stderr, flush=True)
    
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        try:
            os.makedirs(csv_dir, exist_ok=True)
            print(f"[SYNTAX] Создана директория: {csv_dir}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[SYNTAX] ⚠️ Ошибка создания директории: {e}", file=sys.stderr, flush=True)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'relation', 'weight'])
            for source, target, relation, weight in edges_full:
                writer.writerow([source, target, relation, f"{weight:.2f}"])
        print(f"[SYNTAX] ✓ CSV сохранён ({len(edges_full)} рёбер)", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[SYNTAX] ✗ Ошибка CSV: {e}", file=sys.stderr, flush=True)
        return False

def create_html_visualization(edges: List[Tuple[str, str, str, float]], nodes: List[str], html_path: str) -> bool:
    """Creates HTML visualization of the graph"""
    print(f"[SYNTAX] Создаю HTML визуализацию: {html_path}", file=sys.stderr, flush=True)
    
    html_dir = os.path.dirname(html_path)
    if html_dir and not os.path.exists(html_dir):
        try:
            os.makedirs(html_dir, exist_ok=True)
        except Exception as e:
            print(f"[SYNTAX] ⚠️ Ошибка создания директории: {e}", file=sys.stderr, flush=True)
    
    try:
        nodes_data = []
        for i, node in enumerate(nodes):
            nodes_data.append({
                "id": i,
                "label": node,
                "title": node,
                "size": 30,
                "color": "#e74c3c"
            })
        
        edges_data = []
        node_map = {n: i for i, n in enumerate(nodes)}
        
        for source, target, relation, weight in edges:
            edges_data.append({
                "from": node_map[source],
                "to": node_map[target],
                "label": relation[:15],
                "weight": float(weight),
                "title": f"{relation} (вес: {weight:.2f})",
                "color": {"color": "#e74c3c"},
                "arrows": "to"
            })
        
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Синтаксический граф</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        #network {{ width: 100%; height: 100vh; border: 1px solid lightgray; }}
        .info {{ position: absolute; top: 10px; left: 10px; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); z-index: 10; }}
    </style>
</head>
<body>
    <div class="info">
        <h3>Синтаксический граф</h3>
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
        print(f"[SYNTAX] ✓ HTML сохранён", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[SYNTAX] ✗ Ошибка HTML: {e}", file=sys.stderr, flush=True)
        return False

def main():
    """Main function - ✅ ИСПРАВЛЕНИЕ 6: Читаем из файла, а не из stdin"""
    try:
        if len(sys.argv) < 2:
            print("[SYNTAX] ✗ ОШИБКА: укажите путь к JSON файлу", file=sys.stderr, flush=True)
            print("[SYNTAX] Использование: python graph_builder_syntax.py <input.json>", file=sys.stderr, flush=True)
            sys.exit(1)
        
        json_file = sys.argv[1]
        print(f"[SYNTAX] Читаю JSON из файла: {json_file}", file=sys.stderr, flush=True)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            json_input = json.load(f)
        
        print(f"[SYNTAX] ✓ JSON загружен успешно", file=sys.stderr, flush=True)
        
    except FileNotFoundError:
        print(f"[SYNTAX] ✗ ОШИБКА: файл не найден: {json_file}", file=sys.stderr, flush=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[SYNTAX] ✗ ОШИБКА парсинга JSON: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[SYNTAX] ✗ ОШИБКА чтения: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    text = json_input.get('text', '')
    graph_type = json_input.get('graph_type', 'syntax')
    output_csv = json_input.get('output_csv', './output.csv')
    output_html = json_input.get('output_html', './output.html')
    
    if not text:
        print("[SYNTAX] ✗ ОШИБКА: текст пуст", file=sys.stderr, flush=True)
        sys.exit(1)
    
    print(f"[SYNTAX] Обрабатываю текст ({len(text)} символов)...", file=sys.stderr, flush=True)
    
    edges_viz, nodes, edges_full = extract_syntax_graph(text)
    
    save_csv(edges_full, output_csv)
    create_html_visualization(edges_viz, nodes, output_html)
    
    print("[SYNTAX] ✓ Готово!", file=sys.stderr, flush=True)
    print(json.dumps({
        "status": "success",
        "nodes_count": len(nodes),
        "edges_count": len(edges_viz),
        "csv_file": output_csv,
        "html_file": output_html
    }))

if __name__ == "__main__":
    main()