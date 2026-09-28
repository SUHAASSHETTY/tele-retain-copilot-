"""Build (or refresh) the local Chroma index over data/policy_corpus with MiniLM embeddings.

Run: python -m scripts.build_policy_index
"""

from __future__ import annotations

from src.tools.rag_tool import build_index

if __name__ == "__main__":
    info = build_index()
    print(f"indexed {info['chunks']} clauses into '{info['collection']}' "
          f"(fingerprint {info['fingerprint']}) at {info['path']}")
