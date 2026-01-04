# -*- coding: utf-8 -*-
import json
import sys
import io
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv

warnings.filterwarnings('ignore', category=UserWarning, module='confection')
os.environ['SPACY_MAX_LENGTH'] = '2000000'

import numpy as np

try:
    import spacy
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    print(json.dumps({"error": "Установите модель spaCy: python -m spacy download ru_core_news_sm"}, ensure_ascii=False), flush=True)
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Ошибка загрузки spaCy: {e}"}, ensure_ascii=False), flush=True)
    sys.exit(1)

try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
except Exception as e:
    print(json.dumps({"error": f"Ошибка загрузки pymorphy3: {e}"}, ensure_ascii=False), flush=True)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
except Exception as e:
    print(json.dumps({"error": f"Ошибка загрузки SentenceTransformer: {e}"}, ensure_ascii=False), flush=True)
    sys.exit(1)

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print(json.dumps({"error": f"Ошибка загрузки sklearn: {e}"}, ensure_ascii=False), flush=True)
    sys.exit(1)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DIST_THRESHOLD = 7.0
PREDICATE_SIMILARITY_THRESHOLD = 0.55
PRONOUNS = {"он", "она", "оно", "они", "я", "ты", "мы", "вы", "его", "её", "их"}
STOP_POS = {"PRON", "PART", "PUNCT", "DET", "CCONJ", "SCONJ", "SYM"}
STOP_TOKENS = {"-", "—", "–"}
STOP_LEMMAS = {
    "он", "она", "оно", "они", "я", "ты", "мы", "вы",
    "который", "какой", "какая", "какое", "какие",
    "что", "это", "там", "тут", "тот", "те", "такой",
    "сам", "сама", "самый"
}

WH_WORDS = {
    "кто", "что", "кем", "чем", "кого", "чему", "кому", "когда",
    "какой", "какая", "какое", "какие", "каким", "какими", "какого"
}

OBJECT_LIKE_DEPS = {"obj", "iobj", "obl", "xcomp", "acomp", "attr", "ccomp", "pobj"}

ACTION_KEYWORDS = {
    "дела", "делал", "делала", "делали", "делать", "сдела", "сделал", "сделала",
    "занимал", "занимала", "занимался", "занималась", "делаю", "делаешь", "делаем", "делаете",
    "произошло", "произошла", "произошли", "происходит", "происходила", "произойдёт",
    "было", "была", "были", "будет"
}

def normalize(text) -> str:
    """Нормализация текста в леммы"""
    if not text:
        return ""
    if hasattr(text, "text"):
        text = text.text
    
    lemmas = []
    for word in str(text).split():
        try:
            lemma = morph.parse(word)[0].normal_form
            lemmas.append(lemma)
        except:
            lemmas.append(word.lower())
    return " ".join(lemmas)

def get_embedding(text) -> Optional[np.ndarray]:
    """Получить эмбеддинг текста"""
    if not text:
        return None
    try:
        vec = embedder.encode([str(text)])
        if vec is None or len(vec) == 0:
            return None
        return np.array(vec[0], dtype=float)
    except Exception as e:
        print(f"[EMBED_ERR] {e}", file=sys.stderr, flush=True)
        return None

def answer_question(question: str, triple_meta: List[Dict]) -> Dict:
    """Поиск ответа на вопрос по базе триплетов графа"""
    print(f"[SEARCH] Вопрос: {question}", file=sys.stderr, flush=True)
    print(f"[SEARCH] Триплетов в графе: {len(triple_meta)}", file=sys.stderr, flush=True)

    if not triple_meta:
        print("[SEARCH] Граф пуст - нет триплетов", file=sys.stderr, flush=True)
        return {"status": "Граф пуст", "answers": []}

    q_text_lower = question.lower()

    action_question = any(keyword in q_text_lower for keyword in ACTION_KEYWORDS)

    print(f"[SEARCH] Action question: {action_question}", file=sys.stderr, flush=True)
    print(f"[SEARCH] Keywords found: {[k for k in ACTION_KEYWORDS if k in q_text_lower]}", file=sys.stderr, flush=True)

    answers = {}

    if action_question:
        print(f"[SEARCH] Ищем действия в графе", file=sys.stderr, flush=True)
        for tri in triple_meta:
            word = tri.get("s", "").strip()
            rel = tri.get("p", "").strip()

            if not word:
                continue

            answers[word] = answers.get(word, 0) + 1

            print(f"[SEARCH] Найдено действие: {word} (rel: {rel}, вес: {answers[word]})", file=sys.stderr, flush=True)

        if answers:
            sorted_answers = sorted(answers.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"[SEARCH] ✓ Найдено {len(sorted_answers)} действий", file=sys.stderr, flush=True)
            return {"status": " Найден ответ", "answers": sorted_answers}

    print(f"[SEARCH] Выполняем общий поиск по графу", file=sys.stderr, flush=True)

    for tri in triple_meta:
        s = tri.get("s", "").strip()
        p = tri.get("p", "").strip()
        o = tri.get("o", "").strip()

        if not s or not p or not o:
            continue

        key = f"{s} --{p}--> {o}"
        answers[key] = answers.get(key, 0) + 1

    if answers:
        sorted_answers = sorted(answers.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"[SEARCH] ✓ Найдено {len(sorted_answers)} результатов в графе", file=sys.stderr, flush=True)
        return {"status": " Найден ответ", "answers": sorted_answers}

    print("[SEARCH] Ответы не найдены", file=sys.stderr, flush=True)
    return {"status": "Ответы не найдены", "answers": []}

def process_kg_request(json_input: str) -> str:
    """Обработка JSON запроса с графом и вопросом"""
    print(f"[INPUT] Получен JSON запрос, размер: {len(json_input)} символов", file=sys.stderr, flush=True)

    try:
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Ошибка парсинга JSON: {e}", file=sys.stderr, flush=True)
        return json.dumps({"error": f"JSON error: {e}"}, ensure_ascii=False)

    if "graph" not in data:
        print("[ERROR] Отсутствует поле 'graph'", file=sys.stderr, flush=True)
        return json.dumps({"error": "Missing field: graph"}, ensure_ascii=False)

    if "question" not in data:
        print("[ERROR] Отсутствует поле 'question'", file=sys.stderr, flush=True)
        return json.dumps({"error": "Missing field: question"}, ensure_ascii=False)

    graph = data["graph"]
    question = data["question"]

    print(f"[INPUT] Граф получен, edges: {len(graph.get('edges', []))}", file=sys.stderr, flush=True)

    edges = []
    if "edges" in graph and isinstance(graph["edges"], list):
        print(f"[INPUT] Граф передан как JSON, edges: {len(graph['edges'])}", file=sys.stderr, flush=True)
        edges = graph["edges"]

    elif "csv_file" in graph and isinstance(graph["csv_file"], str):
        print(f"[INPUT] Граф передан как CSV файл: {graph['csv_file']}", file=sys.stderr, flush=True)
        csv_path = graph["csv_file"]
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None) 
                print(f"[INPUT] CSV header: {header}", file=sys.stderr, flush=True)
                for i, row in enumerate(reader):
                    if len(row) < 3:
                        continue
                    source = row[0].strip()
                    target = row[1].strip()
                    relation = row[2].strip()
                    weight = float(row[3]) if len(row) > 3 else 1.0
                    edges.append({
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "weight": weight
                    })
                print(f"[INPUT] Загружено {len(edges)} рёбер из CSV", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[ERROR] Ошибка чтения CSV: {e}", file=sys.stderr, flush=True)
            return json.dumps({"error": f"CSV read error: {e}"}, ensure_ascii=False)
    else:
        print("[ERROR] Граф в неподдерживаемом формате", file=sys.stderr, flush=True)
        return json.dumps({"error": "Graph format not supported"}, ensure_ascii=False)

    print(f"[INPUT] Всего рёбер: {len(edges)}", file=sys.stderr, flush=True)

    triple_meta = []
    pos_tags = {}

    for i, edge in enumerate(edges):
        try:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            relation = str(edge.get("relation", "")).strip()
            weight = float(edge.get("weight", 1.0))

            if not source or not target or not relation:
                print(f"[WARN] Пропускаем ребро #{i}: пустые значения", file=sys.stderr, flush=True)
                continue

            source = normalize(source)
            target = normalize(target)
            relation = relation.strip()

            triple_meta.append({
                "s": source,
                "p": relation,
                "o": target,
                "rel": relation,
                "prep": None,
                "s_pos": pos_tags.get(source, "NOUN"),
                "p_pos": pos_tags.get(relation, "VERB"),
                "o_pos": pos_tags.get(target, "NOUN"),
                "weight": weight
            })
        except Exception as e:
            print(f"[WARN] Ошибка обработки ребра #{i}: {e}", file=sys.stderr, flush=True)
            continue

    print(f"[INPUT] Обработано триплетов: {len(triple_meta)}", file=sys.stderr, flush=True)

    result = answer_question(question, triple_meta)

    response = {
        "question": question,
        "status": result["status"],
        "answers": [
            {"answer": ans[0], "score": float(ans[1])}
            for ans in result["answers"][:10]  # Топ 10
        ]
    }

    print(f"[OUTPUT] Возвращаем {len(response['answers'])} ответов", file=sys.stderr, flush=True)
    return json.dumps(response, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if os.path.isfile(arg):
                with open(arg, "r", encoding="utf-8") as f:
                    json_input = f.read()
            else:
                json_input = arg
        else:
            json_input = sys.stdin.read()

        result = process_kg_request(json_input)
        print(result, flush=True)

    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        error_response = {"error": str(e), "status": "error", "answers": []}
        print(json.dumps(error_response, ensure_ascii=False), flush=True)
        sys.exit(1)