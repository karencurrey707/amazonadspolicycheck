#!/usr/bin/env python3
"""
Ask questions of the Amazon Ads policy RAG.

This is the "query" half of the project. load_amazon_ads_policies.py stored the chunks;
this script finds the chunks that best match your question and, optionally, asks a
Nebius chat model to write an answer from them.

Setup (same Terminal window and keys as the loader)
    export NEBIUS_API_KEY="..."
    export PINECONE_API_KEY="..."

Usage
    python3 ask_policy_rag.py                      # interactive: type a question, press Enter
    python3 ask_policy_rag.py "Can I advertise alcohol to people under 21?"
    python3 ask_policy_rag.py --splitter recursive "..."   # search one chunking strategy only
    python3 ask_policy_rag.py --answer "..."       # also draft an answer with a chat model

By default it searches every chunking strategy you loaded (recursive, token, semantic)
and shows the top matches for each, side by side, so you can compare how well each one retrieves.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSIONS = 1024
PINECONE_INDEX = "amazon-ads-policies"
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
# Used only with --answer. Check the model list in the Nebius console and change it if this ID is not available.
CHAT_MODEL = os.environ.get("NEBIUS_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

STRATEGIES = ["recursive", "token", "semantic"]


def pick_namespaces(all_namespaces: list[str], splitter: str) -> list[str]:
    """Match namespaces by strategy name, so it works whatever chunk sizes were used when loading."""
    wanted = STRATEGIES if splitter == "all" else [splitter]
    picked = []
    for strategy in wanted:
        picked.extend(sorted(ns for ns in all_namespaces if ns.startswith(strategy + "-")))
    return picked


def search(index, vector: list[float], namespace: str, top_k: int) -> list[dict]:
    res = index.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)
    matches = []
    for m in res.matches:
        meta = m.metadata or {}
        matches.append(
            {
                "score": float(m.score),
                "category": meta.get("category", ""),
                "source_url": meta.get("source_url", ""),
                "text": meta.get("text", ""),
            }
        )
    return matches


def build_prompt(question: str, matches: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] ({m['category']})\n{m['text']}" for i, m in enumerate(matches, start=1)
    )
    system = (
        "You answer questions about Amazon Ads advertising policies. Use only the numbered excerpts "
        "provided. Cite the excerpt numbers you used, like [1]. If the excerpts do not contain the "
        "answer, say so plainly instead of guessing."
    )
    user = f"Excerpts:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def show_matches(matches: list[dict], width: int) -> None:
    if not matches:
        print("  (no matches)")
        return
    for i, m in enumerate(matches, start=1):
        snippet = " ".join(m["text"].split())
        if len(snippet) > width:
            snippet = snippet[:width].rstrip() + "..."
        print(f"  {i}. score {m['score']:.3f}  |  {m['category']}")
        if m["source_url"]:
            print(f"     {m['source_url']}")
        print(textwrap.fill(snippet, width=96, initial_indent="     ", subsequent_indent="     "))
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Ask questions of the Amazon Ads policy RAG.")
    p.add_argument("question", nargs="*", help="your question (leave blank for interactive mode)")
    p.add_argument("--splitter", choices=STRATEGIES + ["all"], default="all")
    p.add_argument("--top-k", type=int, default=3, help="matches to show per strategy")
    p.add_argument("--chars", type=int, default=400, help="characters of each chunk to display")
    p.add_argument("--answer", action="store_true", help="also draft an answer with a Nebius chat model")
    p.add_argument("--chat-model", default=CHAT_MODEL)
    args = p.parse_args()

    nebius_key = os.environ.get("NEBIUS_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not nebius_key or not pinecone_key:
        sys.exit("Set NEBIUS_API_KEY and PINECONE_API_KEY first (see the top of this file).")

    from openai import OpenAI
    from pinecone import Pinecone

    client = OpenAI(api_key=nebius_key, base_url=NEBIUS_BASE_URL)
    index = Pinecone(api_key=pinecone_key).Index(PINECONE_INDEX)

    stats = index.describe_index_stats()
    namespaces = pick_namespaces(list(stats.namespaces.keys()), args.splitter)
    if not namespaces:
        sys.exit(f"No stored chunks found for '{args.splitter}'. Run the loader for that strategy first.")
    print("Searching: " + ", ".join(namespaces))

    def ask(question: str) -> None:
        vector = client.embeddings.create(
            model=EMBEDDING_MODEL, input=[question], dimensions=EMBEDDING_DIMENSIONS
        ).data[0].embedding
        for ns in namespaces:
            matches = search(index, vector, ns, args.top_k)
            print(f"\n=== {ns} ===")
            show_matches(matches, args.chars)
            if args.answer and matches:
                reply = client.chat.completions.create(
                    model=args.chat_model, messages=build_prompt(question, matches), temperature=0
                )
                print("  Drafted answer:")
                print(textwrap.fill(reply.choices[0].message.content.strip(), width=96,
                                    initial_indent="  ", subsequent_indent="  "))
                print()

    if args.question:
        ask(" ".join(args.question))
        return
    print("Type a question and press Enter. Press Enter on a blank line to quit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    main()
