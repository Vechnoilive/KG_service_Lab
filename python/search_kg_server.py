import json
import sys
import io
import os
import warnings
from collections import defaultdict, deque
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
    "какой", "какая", "какое", "какие", "каким", "какими", "какого",
    "где", "куда", "откуда", "зачем", "почему", "как", "сколько"
}

REL_NSUBJ = "syntax:nsubj"
REL_OBL = "syntax:obl"
REL_ADVMOD = "syntax:advmod"
REL_ATTR = "syntax:attr"
REL_OBJ = "syntax:obj"
REL_IOBJ = "syntax:iobj"
REL_POBJ = "syntax:pobj"

CONNECT_RELS = {"syntax:conj", "syntax:xcomp", "syntax:ccomp", "syntax:advcl", "syntax:parataxis"}
COMPLEMENT_RELS = {REL_OBJ, REL_IOBJ, REL_OBL, REL_POBJ, REL_ATTR}

ACTION_KEYWORDS = {
    "дела", "делал", "делала", "делали", "делать", "сдела", "сделал", "сделала",
    "занимал", "занимала", "занимался", "занималась",
    "произошло", "произошла", "произошли", "происходит", "происходила", "произойдёт"
}
COPULA_FORMS = {"был", "была", "были", "будет", "есть", "является", "являлся", "являлась", "значит", "называется"}

_NORM_CACHE: Dict[str, str] = {}
_POS_CACHE: Dict[str, str] = {}
_EMB_CACHE: Dict[str, np.ndarray] = {}


def normalize(text) -> str:
    if not text:
        return ""
    if hasattr(text, "text"):
        text = text.text
    s = str(text).strip()
    if not s:
        return ""
    cached = _NORM_CACHE.get(s)
    if cached is not None:
        return cached

    lemmas = []
    for word in s.split():
        w = word.strip()
        if not w:
            continue
        try:
            lemma = morph.parse(w)[0].normal_form
            lemmas.append(lemma)
        except Exception:
            lemmas.append(w.lower())

    out = " ".join(lemmas).strip()
    _NORM_CACHE[s] = out
    return out


def _word_pos(word: str) -> str:
    if not word:
        return ""
    w = word.strip().lower()
    if not w:
        return ""
    cached = _POS_CACHE.get(w)
    if cached is not None:
        return cached
    try:
        tag = morph.parse(w)[0].tag
        pos = str(tag.POS) if tag and tag.POS else ""
    except Exception:
        pos = ""
    _POS_CACHE[w] = pos
    return pos


def _is_verb(word: str) -> bool:
    pos = _word_pos(word)
    return pos in {"VERB", "INFN"}


def get_embedding(text) -> Optional[np.ndarray]:
    if not text:
        return None
    key = str(text).strip()
    if not key:
        return None
    cached = _EMB_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        vec = embedder.encode([key])
        if vec is None or len(vec) == 0:
            return None
        arr = np.array(vec[0], dtype=float)
        _EMB_CACHE[key] = arr
        return arr
    except Exception as e:
        print(f"[EMBED_ERR] {e}", file=sys.stderr, flush=True)
        return None


def _get_q_embedding(*texts: str) -> Optional[np.ndarray]:
    for t in texts:
        emb = get_embedding(t)
        if emb is not None:
            return emb
    return None


def _safe_cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        return float(cosine_similarity([a], [b])[0][0])
    except Exception:
        return 0.0


def _question_first_word(question: str) -> Optional[str]:
    q = (question or "").strip().lower()
    if not q:
        return None
    return q.split()[0] if q.split() else None


def _is_action_question(question: str) -> bool:
    q = (question or "").lower()
    if any(k in q for k in ACTION_KEYWORDS):
        return True
    try:
        doc = nlp(question)
        lemmas = {t.lemma_.lower() for t in doc if t.lemma_}
        if "что" in {t.text.lower() for t in doc} and ("делать" in lemmas or "заниматься" in lemmas):
            return True
    except Exception:
        pass
    return False


def _looks_like_copula_question(question: str) -> bool:
    q = (question or "").lower()
    if any(w in q.split() for w in COPULA_FORMS):
        return True
    try:
        doc = nlp(question)
        lemmas = [t.lemma_.lower() for t in doc if t.lemma_]
        return "быть" in lemmas or "являться" in lemmas
    except Exception:
        return False


def _extract_agent_from_question(question: str) -> Optional[str]:
    q = (question or "").strip()
    if not q:
        return None
    try:
        doc = nlp(q)
    except Exception:
        return None

    cand = [t for t in doc if t.dep_.startswith("nsubj") and t.pos_ in {"NOUN", "PROPN"}]
    if not cand:
        cand = []
        for t in doc:
            if t.pos_ not in {"NOUN", "PROPN"}:
                continue
            lemma = (t.lemma_ or t.text).lower().strip()
            if not lemma or lemma in WH_WORDS or lemma in STOP_LEMMAS:
                continue
            cand.append(t)

    if not cand:
        return None
    agent = (cand[-1].lemma_ or cand[-1].text).lower().strip()
    return normalize(agent)


def _parse_main_verb_and_constraints(question: str) -> Tuple[Optional[str], List[str]]:
    q = (question or "").strip()
    if not q:
        return None, []
    try:
        doc = nlp(q)
    except Exception:
        return None, []

    root = None
    for t in doc:
        if t.dep_ == "ROOT":
            root = t
            break

    verb = None
    if root is not None:
        if root.pos_ in {"VERB", "AUX"}:
            verb = normalize((root.lemma_ or root.text).lower())
        else:
            for t in doc:
                if t.pos_ in {"VERB", "AUX"}:
                    verb = normalize((t.lemma_ or t.text).lower())
                    break

    constraints = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        lemma = (t.lemma_ or t.text).lower().strip()
        if not lemma or lemma in WH_WORDS or lemma in STOP_LEMMAS:
            continue
        if t.dep_ in {"advmod", "obl", "nmod"} or t.pos_ in {"ADV"}:
            constraints.append(normalize(lemma))

    if verb:
        constraints = [c for c in constraints if c != verb]

    seen = set()
    uniq = []
    for c in constraints:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)

    return verb, uniq


def _build_maps(triple_meta: List[Dict]) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, str, float]]], Dict[str, List[Tuple[str, str, float]]]]:
    subj_map = defaultdict(list)
    mod_map = defaultdict(list)
    verb_neigh = defaultdict(list)

    for tri in triple_meta:
        s = (tri.get("s") or "").strip()
        o = (tri.get("o") or "").strip()
        p = (tri.get("p") or "").strip().lower()
        w = float(tri.get("weight", 1.0) or 1.0)
        if not s or not o or not p:
            continue

        if REL_NSUBJ in p:
            subj_map[s].append((o, w))

        if REL_ADVMOD in p or REL_OBL in p:
            mod_map[s].append((o, p, w))

        if any(p.startswith(r) for r in CONNECT_RELS) and _is_verb(s) and _is_verb(o):
            verb_neigh[s].append((o, p, w))
            verb_neigh[o].append((s, p, w))

    return subj_map, mod_map, verb_neigh


def _w_norm(x: float) -> float:
    x = max(0.0, min(float(x), 12.0))
    return x / 12.0


def _answer_actions(question: str, triple_meta: List[Dict]) -> Dict:
    agent = _extract_agent_from_question(question)
    print(f"[SEARCH] Action-mode agent: {agent}", file=sys.stderr, flush=True)
    if not agent:
        return {"status": "Не удалось выделить субъекта", "answers": []}

    verb_neighbors = defaultdict(list)
    verb_complements = defaultdict(list)

    for tri in triple_meta:
        s = (tri.get("s") or "").strip()
        o = (tri.get("o") or "").strip()
        p = (tri.get("p") or "").strip().lower()
        w = float(tri.get("weight", 1.0) or 1.0)
        if not s or not o or not p:
            continue

        if any(p.startswith(r) for r in CONNECT_RELS) and _is_verb(s) and _is_verb(o):
            verb_neighbors[s].append((o, p, w))
            verb_neighbors[o].append((s, p, w))

        if any(r in p for r in COMPLEMENT_RELS) and _is_verb(s) and (not _is_verb(o)):
            verb_complements[s].append((o, p, w))

    seeds = []
    for tri in triple_meta:
        s = (tri.get("s") or "").strip()
        o = (tri.get("o") or "").strip()
        p = (tri.get("p") or "").strip().lower()
        w = float(tri.get("weight", 1.0) or 1.0)
        if not s or not o or not p:
            continue
        if o == agent and (REL_NSUBJ in p) and _is_verb(s):
            seeds.append((s, w))

    if not seeds:
        return {"status": "Действия субъекта не найдены", "answers": []}

    max_depth = 2
    best_score = {}
    q = deque()

    for v, w in seeds:
        sc = 1.0 * (0.7 + 0.3 * _w_norm(w))
        if sc > best_score.get(v, 0.0):
            best_score[v] = sc
            q.append((v, 0, sc))

    while q:
        v, d, sc = q.popleft()
        if d >= max_depth:
            continue
        for u, rel, w in verb_neighbors.get(v, []):
            new_sc = sc * 0.75 * (0.75 + 0.25 * _w_norm(w))
            if new_sc > best_score.get(u, 0.0):
                best_score[u] = new_sc
                q.append((u, d + 1, new_sc))

    def best_complement(verb: str) -> Optional[str]:
        cand = []
        for obj, rel, w in verb_complements.get(verb, []):
            o = (obj or "").strip()
            if not o:
                continue
            if o == agent:
                continue
            if o in PRONOUNS or o in STOP_LEMMAS:
                continue
            cand.append((o, float(w)))
        if not cand:
            return None
        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[0][0]

    actions = []
    for v, sc in best_score.items():
        comp = best_complement(v)
        if comp:
            actions.append((f"{v} {comp}", float(sc)))
        else:
            actions.append((v, float(sc)))

    q_emb = _get_q_embedding(normalize(question), question)
    out = []
    for text, sc in actions:
        sim = _safe_cos(q_emb, get_embedding(text))
        out.append((text, float(sc * 0.85 + sim * 0.15)))

    best_by_text = {}
    for t, s in out:
        if t not in best_by_text or s > best_by_text[t]:
            best_by_text[t] = s

    top = sorted(best_by_text.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"status": "Найден ответ", "answers": top}


def _answer_who(question: str, triple_meta: List[Dict]) -> Dict:
    verb_q, constraints = _parse_main_verb_and_constraints(question)
    subj_map, mod_map, verb_neigh = _build_maps(triple_meta)

    verb_score = defaultdict(float)

    for v, mods in mod_map.items():
        if not _is_verb(v):
            continue
        for m, rel, w in mods:
            if constraints and m in constraints:
                verb_score[v] += 1.0 + 0.4 * _w_norm(w)

    if verb_q:
        verb_score[verb_q] += 0.9
        qv_emb = get_embedding(verb_q)
        if qv_emb is not None:
            for v in list(subj_map.keys())[:400]:
                if not _is_verb(v):
                    continue
                sim = _safe_cos(qv_emb, get_embedding(v))
                if sim >= 0.55:
                    verb_score[v] += 0.5 * sim

    seeds = sorted(verb_score.items(), key=lambda x: x[1], reverse=True)[:5]
    for v, sc in seeds:
        for u, rel, w in verb_neigh.get(v, []):
            if not _is_verb(u):
                continue
            verb_score[u] = max(verb_score[u], sc * 0.7 + 0.2 * _w_norm(w))

    if not verb_score:
        return {"status": "Ответы не найдены", "answers": []}

    answers = []
    for v, vsc in verb_score.items():
        for subj, w in subj_map.get(v, []):
            s = (subj or "").strip()
            if not s:
                continue
            if s in PRONOUNS or s in STOP_LEMMAS:
                continue
            bonus = 0.0
            if constraints:
                for m, rel, ww in mod_map.get(v, []):
                    if m in constraints:
                        bonus += 0.25
                        break
            score = float(vsc + 0.6 + bonus + 0.15 * _w_norm(w))
            answers.append((s, score))

    if not answers:
        return {"status": "Ответы не найдены", "answers": []}

    best = {}
    for a, sc in answers:
        if a not in best or sc > best[a]:
            best[a] = sc

    top = sorted(best.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"status": "Найден ответ", "answers": top}


def answer_question(question: str, triple_meta: List[Dict]) -> Dict:
    print(f"[SEARCH] Вопрос: {question}", file=sys.stderr, flush=True)
    print(f"[SEARCH] Триплетов в графе: {len(triple_meta)}", file=sys.stderr, flush=True)

    if not triple_meta:
        return {"status": "Граф пуст", "answers": []}

    wh = _question_first_word(question)

    if wh == "кто":
        return _answer_who(question, triple_meta)

    if _is_action_question(question):
        return _answer_actions(question, triple_meta)

    q_emb = _get_q_embedding(normalize(question), question)

    answers = []
    for tri in triple_meta:
        s = (tri.get("s") or "").strip()
        p = (tri.get("p") or "").strip()
        o = (tri.get("o") or "").strip()
        w = float(tri.get("weight", 1.0) or 1.0)
        if not s or not p or not o:
            continue

        text = f"{s} --{p}--> {o}"
        sim = _safe_cos(q_emb, get_embedding(f"{s} {p} {o}"))
        score = 0.85 * sim + 0.15 * _w_norm(w)
        answers.append((text, float(score)))

    answers.sort(key=lambda x: x[1], reverse=True)
    top = answers[:10]
    status = "Найден ответ" if top else "Ответы не найдены"
    return {"status": status, "answers": top}


def process_kg_request(json_input: str) -> str:
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
            for ans in result["answers"][:10]
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
