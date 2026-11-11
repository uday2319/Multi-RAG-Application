import json
import re
from pathlib import Path


class PaperRouter:
    """
    Detects which papers a question names, using the aliases in
    data/papers/papers.json, e.g. "How do REALM and RAG differ?"
    -> ["realm_guu2020.pdf", "rag_lewis2020.pdf"].
    """

    def __init__(self, catalog_path):

        catalog_path = Path(catalog_path)

        self.catalog = (
            json.loads(catalog_path.read_text(encoding="utf-8"))
            if catalog_path.exists() else {}
        )

        # (alias, source) pairs, longest alias first so that
        # "Self-RAG" is matched before "RAG"
        self.aliases = sorted(
            (
                (alias, source)
                for source, info in self.catalog.items()
                for alias in info.get("aliases", [])
            ),
            key=lambda pair: len(pair[0]),
            reverse=True
        )

    def detect(self, query):
        """Return the sources named in the query, in order of appearance."""

        text = query
        found = []  # (position, source)

        for alias, source in self.aliases:

            # Word boundaries that also treat "-" as part of a word,
            # so "RAG" does not match inside "Self-RAG"
            pattern = re.compile(
                r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])",
                re.IGNORECASE
            )

            for match in pattern.finditer(text):
                found.append((match.start(), source))

            # Blank out matches so shorter aliases cannot re-match them
            text = pattern.sub(lambda m: " " * len(m.group()), text)

        ordered = []
        for _, source in sorted(found):
            if source not in ordered:
                ordered.append(source)

        return ordered

    def short_name(self, source):
        return self.catalog.get(source, {}).get("short", source)
