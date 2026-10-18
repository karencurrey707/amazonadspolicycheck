#!/usr/bin/env python3
"""
Load Amazon Ads Policies into Pinecone.

A Python port of the n8n workflow "Workflow 1 - Load Amazon Ads Policies".

Pipeline
    1. Read the list of Google Docs in a public Drive folder.
    2. Download each doc as plain text and clean it.
    3. Chunk it with one of three strategies: recursive, token, or semantic.
    4. Embed each chunk with a Nebius embedding model.
    5. Upsert the vectors into Pinecone, in a namespace named after the strategy.

Setup
    pip install -r requirements.txt
    export NEBIUS_API_KEY="..."      # from the Nebius Token Factory console
    export PINECONE_API_KEY="..."    # from the Pinecone console

Usage
    python load_amazon_ads_policies.py --dry-run                 # preview, no embedding, no upload
    python load_amazon_ads_policies.py --splitter recursive      # load 1 doc (the testing default)
    python load_amazon_ads_policies.py --splitter all --docs 0   # all three strategies, every doc

Notes
    * Settings below match the n8n workflow. Change them here or with command line flags.
    * The folder must be shared as "Anyone with the link".
    * Chunk sizes follow the workflow: recursive counts characters, token counts tokens,
      and semantic uses chunk_size as a maximum character count per chunk.
    * Vector IDs are deterministic, so re-running the script overwrites instead of duplicating.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Settings (same values as the "Settings" node in the n8n workflow)
# ---------------------------------------------------------------------------
SETTINGS = {
    "drive_folder_id": "1aFvz5DBugHBQL10xcSmw19WNUMeyu3KD",
    "docs_to_process": 1,           # 0 means every doc found
    "splitter_type": "semantic",    # recursive | token | semantic | all
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "embedding_dimensions": 1024,
    "semantic_threshold": 0.5,
    # The old workflow used https://api.studio.nebius.com/v1 (the former AI Studio address).
    # Nebius Token Factory is the current name. Override with NEBIUS_BASE_URL if needed.
    "nebius_base_url": os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
    "pinecone_index": "amazon-ads-policies",
    "pinecone_cloud": "aws",
    "pinecone_region": "us-east-1",
}

# Help Center page IDs, keyed by the first word of each policy doc's name.
POLICY_PAGE_IDS = {
    "prohibited": "GZS8A5XCFQJTS88N",
    "alcohol": "G2MMCHZDCTW37V39",
    "animal": "GPHDMTD25DRYVRM7",
    "dating": "G8S3JBS2CQQ5BXS5",
    "charities": "GPRBZ3P2JA2UCU2U",
    "claims": "G6UU66YLSCUYDULQ",
    "employment": "GH44NYF29M6QCYSE",
    "entertainment": "G7NP3SWRVV6UV66V",
    "financial": "GSMUMHF989KULMEQ",
    "gambling": "GYN86Z5STCSCPDLJ",
    "real": "GYZ3NZ7PXJCZXSPT",
    "healthcare": "GPSSMTEAA6AEWY3A",
    "baby": "GFNDKT3FYGDL54VB",
    "pricing": "GKP5TCJWVN5W7AFU",
    "public": "GEZ4XHT8TRXFB5WH",
    "news": "G8LCCRSMMYE6255Z",
    "sports": "GD8U427Z7HJHHJGP",
    "event": "GQ3SJ5QCWCQCKCYQ",
}

EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Steps 1 and 2: find and read the policy docs
# ---------------------------------------------------------------------------
def _decode_html(s: str) -> str:
    return (
        s.replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .strip()
    )


def parse_folder_html(html: str) -> list[dict]:
    """Pull (doc id, doc name) pairs out of the Drive embedded folder view."""
    pattern = re.compile(r'docs\.google\.com/document/d/([\w-]+)[\s\S]*?flip-entry-title">([^<]+)<')
    seen: set[str] = set()
    docs = []
    for match in pattern.finditer(html):
        doc_id = match.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append({"id": doc_id, "name": _decode_html(match.group(2))})
    return docs


def find_policy_docs(folder_id: str) -> list[dict]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    docs = parse_folder_html(resp.text)
    if not docs:
        raise RuntimeError(
            'No Google Docs found. Check the folder ID and that the folder is shared as "Anyone with the link".'
        )
    return docs


def read_doc_text(doc_id: str) -> str:
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def clean_text(raw: str) -> str:
    text = raw.lstrip("﻿")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_url_for(name: str) -> str:
    first_word = name.lower().split(" ")[0]
    page_id = POLICY_PAGE_IDS.get(first_word)
    if not page_id:
        print(f"  warning: no Help Center page ID for '{name}', source_url left blank", file=sys.stderr)
        return ""
    return f"https://advertising.amazon.com/help/{page_id}"


# ---------------------------------------------------------------------------
# Step 3: chunking strategies
# ---------------------------------------------------------------------------
def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


def chunk_token(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


def split_sentences(text: str) -> list[str]:
    """One sentence per entry. Each bullet line counts as its own sentence."""
    sentences = []
    for raw_line in text.split("\n"):
        line = re.sub(r"^\s*[*\-•]\s*", "", raw_line).strip()
        if not line:
            continue
        for part in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(“])', line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return dot / denom if denom else 0.0


def group_by_meaning(
    sentences: list[str], vectors: list[list[float]], threshold: float, max_chars: int
) -> list[dict]:
    """Walk the sentences in order. Start a new chunk when the next sentence is not
    similar enough to the previous one (topic change) or the chunk would get too long."""
    sentences = sentences[: len(vectors)]
    if not sentences:
        return []
    chunks = []
    current = [sentences[0]]
    length = len(sentences[0])
    scores: list[float] = []

    def push(reason: str) -> None:
        chunks.append(
            {
                "text": "\n".join(current),
                "sentence_count": len(current),
                "similarities_inside": scores,
                "ended_because": reason,
            }
        )

    for k in range(1, len(sentences)):
        sim = round(cosine(vectors[k - 1], vectors[k]), 3)
        too_long = length + 1 + len(sentences[k]) > max_chars
        if sim < threshold or too_long:
            push(f"topic change (similarity {sim})" if sim < threshold else "size limit")
            current = [sentences[k]]
            length = len(sentences[k])
            scores = []
        else:
            current.append(sentences[k])
            length += 1 + len(sentences[k])
            scores.append(sim)
    push("end of document")
    return chunks


def chunk_semantic(text: str, embedder, threshold: float, max_chars: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    vectors = embedder(sentences)
    return [c["text"] for c in group_by_meaning(sentences, vectors, threshold, max_chars)]


# ---------------------------------------------------------------------------
# Step 4: Nebius embeddings (OpenAI compatible API)
# ---------------------------------------------------------------------------
def make_embedder(settings: dict):
    from openai import OpenAI

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        sys.exit("NEBIUS_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url=settings["nebius_base_url"])

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = client.embeddings.create(
                model=settings["embedding_model"],
                input=batch,
                dimensions=settings["embedding_dimensions"],
            )
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)
        return vectors

    return embed


# ---------------------------------------------------------------------------
# Step 5: Pinecone
# ---------------------------------------------------------------------------
def get_pinecone_index(settings: dict):
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        sys.exit("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=api_key)
    name = settings["pinecone_index"]
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        print(f"Creating Pinecone index '{name}' ({settings['embedding_dimensions']} dimensions, cosine)...")
        pc.create_index(
            name=name,
            dimension=settings["embedding_dimensions"],
            metric="cosine",
            spec=ServerlessSpec(cloud=settings["pinecone_cloud"], region=settings["pinecone_region"]),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    return pc.Index(name)


def namespace_for(strategy: str, s: dict) -> str:
    if strategy == "semantic":
        return f"semantic-{s['semantic_threshold']}-{s['chunk_size']}"
    return f"{strategy}-{s['chunk_size']}-{s['chunk_overlap']}"


def vector_id(namespace: str, category: str, position: int) -> str:
    raw = f"{namespace}|{category}|{position}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def upsert(index, namespace: str, records: list[dict]) -> None:
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(vectors=records[i : i + UPSERT_BATCH_SIZE], namespace=namespace)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(settings: dict, strategies: list[str], dry_run: bool) -> None:
    print("Finding policy docs...")
    docs = find_policy_docs(settings["drive_folder_id"])
    print(f"Found {len(docs)} docs.")
    limit = settings["docs_to_process"]
    if limit and limit > 0:
        docs = docs[:limit]
        print(f"Processing the first {len(docs)} (docs_to_process = {limit}).")

    prepared = []
    for doc in docs:
        text = clean_text(read_doc_text(doc["id"]))
        prepared.append(
            {"category": doc["name"], "source_url": source_url_for(doc["name"]), "text": text}
        )
        print(f"  read '{doc['name']}' ({len(text):,} characters)")

    embedder = None
    if (not dry_run) or ("semantic" in strategies):
        embedder = make_embedder(settings)
    index = None if dry_run else get_pinecone_index(settings)

    for strategy in strategies:
        namespace = namespace_for(strategy, settings)
        print(f"\nStrategy: {strategy}  (namespace '{namespace}')")
        for doc in prepared:
            if strategy == "recursive":
                chunks = chunk_recursive(doc["text"], settings["chunk_size"], settings["chunk_overlap"])
            elif strategy == "token":
                chunks = chunk_token(doc["text"], settings["chunk_size"], settings["chunk_overlap"])
            else:
                chunks = chunk_semantic(
                    doc["text"], embedder, settings["semantic_threshold"], settings["chunk_size"]
                )
            avg = sum(len(c) for c in chunks) // max(len(chunks), 1)
            print(f"  '{doc['category']}': {len(chunks)} chunks, average {avg} characters")

            if dry_run:
                for n, chunk in enumerate(chunks[:2], start=1):
                    preview = chunk[:160].replace("\n", " ")
                    print(f"      chunk {n}: {preview}...")
                continue

            vectors = embedder(chunks)
            records = []
            for pos, (chunk, vec) in enumerate(zip(chunks, vectors)):
                metadata = {
                    "text": chunk,
                    "category": doc["category"],
                    "source_url": doc["source_url"],
                    "splitter": strategy,
                    "chunk_size": settings["chunk_size"],
                }
                if strategy == "semantic":
                    metadata["semantic_threshold"] = settings["semantic_threshold"]
                records.append(
                    {"id": vector_id(namespace, doc["category"], pos), "values": vec, "metadata": metadata}
                )
            upsert(index, namespace, records)
            print(f"      stored {len(records)} vectors")

    print("\nDone." if not dry_run else "\nDry run finished. Nothing was uploaded.")


def main() -> None:
    p = argparse.ArgumentParser(description="Load Amazon Ads policy docs into Pinecone.")
    p.add_argument("--splitter", choices=["recursive", "token", "semantic", "all"],
                   default=SETTINGS["splitter_type"])
    p.add_argument("--docs", type=int, default=SETTINGS["docs_to_process"],
                   help="how many docs to process (0 = all)")
    p.add_argument("--chunk-size", type=int, default=SETTINGS["chunk_size"])
    p.add_argument("--chunk-overlap", type=int, default=SETTINGS["chunk_overlap"])
    p.add_argument("--threshold", type=float, default=SETTINGS["semantic_threshold"])
    p.add_argument("--model", default=SETTINGS["embedding_model"])
    p.add_argument("--dimensions", type=int, default=SETTINGS["embedding_dimensions"])
    p.add_argument("--dry-run", action="store_true",
                   help="chunk and preview only; no Pinecone upload (semantic still calls Nebius)")
    args = p.parse_args()

    settings = dict(SETTINGS)
    settings.update(
        splitter_type=args.splitter,
        docs_to_process=args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        semantic_threshold=args.threshold,
        embedding_model=args.model,
        embedding_dimensions=args.dimensions,
    )
    strategies = ["recursive", "token", "semantic"] if args.splitter == "all" else [args.splitter]
    run(settings, strategies, args.dry_run)


if __name__ == "__main__":
    main()
