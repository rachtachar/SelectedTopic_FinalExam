"""ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

รวมการค้นหาทั้งแบบ Vector Search (Semantic) และ Keyword/BM25 Search (Lexical)
เพื่อความแม่นยำสูงสุดในภาษาไทย, รหัสเฉพาะ, ตาราง และตัวเลข
"""
import math
import re
from collections import Counter
from core.types import Chunk
from core.utils import cosine_many
from student import settings


def _tokenize(text: str) -> list[str]:
    """แยกคำและ n-gram เพื่อใช้ในการทำ BM25 สำหรับภาษาไทยและภาษาอังกฤษ"""
    text = text.lower()
    # 1. ดึงคำภาษาอังกฤษ ตัวเลข และรหัส เช่น bb-save-26, vip9, 2026
    words = re.findall(r"[a-z0-9_\-\.]+", text)
    tokens: list[str] = list(words)

    # 2. ข้อความภาษาไทย แยกเป็น character 2-grams และ 3-grams
    thai_blocks = re.findall(r"[\u0E00-\u0E7F]+", text)
    for block in thai_blocks:
        n = len(block)
        for i in range(n - 1):
            tokens.append(block[i:i + 2])
        for i in range(n - 2):
            tokens.append(block[i:i + 3])
    return tokens


def _compute_bm25_scores(chunks: list[Chunk], query: str) -> list[float]:
    """คำนวณคะแนน Okapi BM25 ของแต่ละ chunk เทียบกับ query"""
    n = len(chunks)
    if n == 0:
        return []

    doc_tokens = [_tokenize(c.text) for c in chunks]
    doc_lens = [len(dt) for dt in doc_tokens]
    avgdl = sum(doc_lens) / max(n, 1)

    df: Counter = Counter()
    for dt in doc_tokens:
        for t in set(dt):
            df[t] += 1

    q_tokens = _tokenize(query)
    k1 = 1.5
    b = 0.75

    scores: list[float] = []
    for i in range(n):
        dt = doc_tokens[i]
        counts = Counter(dt)
        doc_len = doc_lens[i]
        score = 0.0
        for t in set(q_tokens):
            if t in df:
                # Okapi BM25 IDF
                idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
                tf = counts[t]
                num = tf * (k1 + 1.0)
                denom = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1e-5)))
                score += idf * (num / denom)
        scores.append(score)

    return scores


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)
    """
    if options and settings.QUERY_WITH_OPTIONS:
        opt_text = " ".join(options)
        return f"{question}\n{opt_text}".strip()
    return question.strip() if question.strip() else "query"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย
    """
    if not chunks:
        return []

    target_k = settings.TOP_K if settings.TOP_K else k
    n = len(chunks)

    # 1. คำนวณคะแนน Vector (Cosine)
    has_vectors = query_vector is not None and any(c.vector is not None for c in chunks)
    if has_vectors and query_vector is not None:
        valid_vectors = [c.vector if c.vector is not None else [0.0] * len(query_vector) for c in chunks]
        vec_scores = cosine_many(query_vector, valid_vectors)
    else:
        vec_scores = [0.0] * n

    # 2. คำนวณคะแนน Keyword (BM25)
    hybrid_w = settings.HYBRID_WEIGHT
    if hybrid_w > 0.0 or not has_vectors:
        bm25_scores = _compute_bm25_scores(chunks, query_text)
    else:
        bm25_scores = [0.0] * n

    # 3. Normalization และ Hybrid Fusion
    if not has_vectors:
        # ถ้าไม่มี vector เลย ให้ใช้ BM25 100%
        final_scores = bm25_scores
    elif hybrid_w <= 0.0:
        # ใช้ Vector 100%
        final_scores = vec_scores
    else:
        # ปรับสเกลให้อยู่ในช่วง [0, 1] ก่อนรวม
        min_v, max_v = min(vec_scores), max(vec_scores)
        norm_vec = [(s - min_v) / (max_v - min_v) if max_v > min_v else 0.5 for s in vec_scores]

        min_b, max_b = min(bm25_scores), max(bm25_scores)
        norm_bm25 = [(s - min_b) / (max_b - min_b) if max_b > min_b else 0.0 for s in bm25_scores]

        final_scores = [
            (1.0 - hybrid_w) * nv + hybrid_w * nb
            for nv, nb in zip(norm_vec, norm_bm25)
        ]

    # จับคู่ (chunk, score)
    pairs = list(zip(chunks, final_scores))

    # กรองคะแนนขั้นต่ำ (ถ้ามีการตั้งค่า)
    if settings.MIN_SCORE is not None:
        pairs = [p for p in pairs if p[1] >= settings.MIN_SCORE]

    # เรียงลำดับจากคะแนนมากไปน้อย
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs[:target_k]
