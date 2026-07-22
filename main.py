import csv
import json
import math
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "documents.csv"), newline="", encoding="utf-8") as f:
    DOCUMENTS = list(csv.DictReader(f))

with open(os.path.join(BASE_DIR, "embeddings.json"), encoding="utf-8") as f:
    EMBEDDINGS: Dict[str, List[float]] = json.load(f)

with open(os.path.join(BASE_DIR, "reranker_scores.json"), encoding="utf-8") as f:
    RERANKER_SCORES: Dict[str, Dict[str, float]] = json.load(f)

# Precompute norms once for speed.
DOC_NORMS = {
    doc_id: math.sqrt(sum(x * x for x in vec)) or 1.0
    for doc_id, vec in EMBEDDINGS.items()
}


def to_number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def value_eq(doc_val, cond_val):
    dn, cn = to_number(doc_val), to_number(cond_val)
    if dn is not None and cn is not None:
        return dn == cn
    return str(doc_val) == str(cond_val)


def value_in(doc_val, cond_list):
    dn = to_number(doc_val)
    for v in cond_list:
        vn = to_number(v)
        if dn is not None and vn is not None and dn == vn:
            return True
        if str(doc_val) == str(v):
            return True
    return False


def compare_op(doc_val, cond_val, op):
    dn, cn = to_number(doc_val), to_number(cond_val)
    if dn is None or cn is None:
        return False
    if op == "gte":
        return dn >= cn
    if op == "lte":
        return dn <= cn
    if op == "gt":
        return dn > cn
    if op == "lt":
        return dn < cn
    return False


def doc_matches(doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, cond in (filters or {}).items():
        doc_val = doc.get(key)
        if isinstance(cond, dict):
            for op, val in cond.items():
                if op == "in":
                    if not value_in(doc_val, val):
                        return False
                elif op == "eq":
                    if not value_eq(doc_val, val):
                        return False
                elif op in ("gte", "lte", "gt", "lt"):
                    if not compare_op(doc_val, val, op):
                        return False
                else:
                    return False
        else:
            if not value_eq(doc_val, cond):
                return False
    return True


def cosine_similarity(a: List[float], b: List[float], norm_b: float) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    return dot / (norm_a * norm_b)


class SearchRequest(BaseModel):
    query_id: str = ""
    query_vector: List[float] = []
    top_k: int = 10
    rerank_top_n: int = 5
    filter: Dict[str, Any] = {}


@app.exception_handler(Exception)
async def general_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={"matches": []})


@app.get("/")
def health():
    return {"status": "ok", "service": "vector-search-rerank"}


@app.post("/vector-search")
def vector_search(payload: SearchRequest):
    if not payload.query_vector:
        return {"matches": []}

    filtered = [doc for doc in DOCUMENTS if doc_matches(doc, payload.filter)]
    if not filtered:
        return {"matches": []}

    scored = []
    for doc in filtered:
        doc_id = doc["doc_id"]
        vec = EMBEDDINGS.get(doc_id)
        if vec is None:
            continue
        sim = cosine_similarity(payload.query_vector, vec, DOC_NORMS.get(doc_id, 1.0))
        scored.append((doc_id, sim))

    # Sort by similarity desc, tie-break by lexicographically smaller doc_id.
    scored.sort(key=lambda x: (-x[1], x[0]))
    top_k = scored[: max(payload.top_k, 0)]

    if not top_k:
        return {"matches": []}

    query_rerank_scores = RERANKER_SCORES.get(payload.query_id, {})
    reranked = [
        (doc_id, query_rerank_scores.get(doc_id, 0.0))
        for doc_id, _ in top_k
    ]
    reranked.sort(key=lambda x: (-x[1], x[0]))

    final = [doc_id for doc_id, _ in reranked[: max(payload.rerank_top_n, 0)]]

    return {"matches": final}
