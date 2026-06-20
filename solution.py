import json
import os

import chromadb

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

EMBED_BACKEND = os.environ.get(
    "EMBED_BACKEND", "gemini" if GOOGLE_API_KEY else "local"
)
CHAT_MODEL = "gemini-2.5-flash"
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.json")
COLLECTION = "kb_lab2"


class GeminiEmbedding(chromadb.EmbeddingFunction):
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def name(self):  
        return "gemini-embedding-001"

    def __call__(self, input):
        resp = self.client.models.embed_content(
            model="gemini-embedding-001", contents=list(input)
        )
        return [e.values for e in resp.embeddings]


class LocalEmbedding(chromadb.EmbeddingFunction):
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def name(self):
        return "all-MiniLM-L6-v2"

    def __call__(self, input):
        return self.model.encode(list(input), convert_to_numpy=True).tolist()


def get_embedding_fn():
    if EMBED_BACKEND == "gemini":
        print("Embeddings: Gemini gemini-embedding-001")
        return GeminiEmbedding()
    print("Embeddings: local all-MiniLM-L6-v2")
    return LocalEmbedding()




def build_index():
    with open(KB_PATH) as f:
        kb = json.load(f)

    client = chromadb.Client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    coll = client.create_collection(COLLECTION, embedding_function=get_embedding_fn())

    coll.add(
        ids=[p["id"] for p in kb],
        documents=[p["text"] for p in kb],
        metadatas=[{"source": p["source"]} for p in kb],
    )
    print(f"Indexed {len(kb)} passages into Chroma collection '{COLLECTION}'\n")
    return coll

def retrieve(coll, question, k=3):
    res = coll.query(query_texts=[question], n_results=k)
    return [
        {"id": i, "source": m["source"], "text": d}
        for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])
    ]


def build_prompt(question, passages):
    context = "\n".join(
        f"[{p['source']} | {p['id']}] {p['text']}" for p in passages
    )
    return f"""You answer questions using ONLY the context below.
Rules:
- Use only facts present in the context.
- After each fact, cite its source in the form (source: <source>).
- If the context does not contain the answer, reply exactly:
  "I don't know — the knowledge base doesn't cover this." Do not guess.

Context:
{context}

Question: {question}
Answer:"""


def generate(prompt):
    if not GOOGLE_API_KEY:
        return "[generation skipped: GOOGLE_API_KEY empty — set it to get a real answer]"
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text.strip()


def answer(coll, question, k=3):
    passages = retrieve(coll, question, k)
    prompt = build_prompt(question, passages)
    reply = generate(prompt)
    return passages, reply


def main():
    coll = build_index()

    questions = [
        "How long do I have to get a full refund?",     # answerable -> kb-04
        "How do I reset my password?",                  # answerable -> kb-07
        "What is the company's stock price today?",     # NOT in KB -> must decline
    ]

    for q in questions:
        passages, reply = answer(coll, q)
        print("=" * 70)
        print(f"Q: {q}")
        print("Retrieved sources:")
        for p in passages:
            print(f"  - {p['id']} ({p['source']}): {p['text'][:70]}...")
        print(f"\nAnswer:\n{reply}\n")

    print("#" * 70)
    print("STRETCH: same answerable question with k=1 vs k=3")
    for k in (1, 3):
        _, reply = answer(coll, "How long do I have to get a full refund?", k=k)
        print(f"\n[k={k}] {reply}")


if __name__ == "__main__":
    main()
