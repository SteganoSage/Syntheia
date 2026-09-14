"""Download the knowledge-base corpus listed in data/knowledge_base/sources.yaml.

    python -m notebooks.09a_fetch_sources              # fetch what is missing
    python -m notebooks.09a_fetch_sources --force      # re-fetch everything
    python -m notebooks.09a_fetch_sources --check      # verify URLs, download nothing

The registry is tracked in git and the documents are not, so this script is what
makes the corpus reproducible -- §5 asks for a version-controlled knowledge base,
and a pile of PDFs nobody can rebuild is not that.

Each download also writes the sidecar manifest the loader requires, so provenance
comes from the registry rather than being retyped per document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

KB = REPO / "data/knowledge_base"
REGISTRY = KB / "sources.yaml"

# Several of these hosts reject the default python-requests agent outright --
# NPCI 403s, and some RBI paths behave differently. A browser agent is the
# difference between a corpus and a list of failures.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/pdf,application/xhtml+xml,*/*",
    "Accept-Language": "en-GB,en;q=0.9",
}

# indiacode.nic.in serves the DPDP Act only when a same-site Referer is present
# -- without one it 404s while curl with a referer gets the PDF. Sent per host
# rather than globally, since a wrong Referer is worse than none.
REFERERS = {
    "www.indiacode.nic.in": "https://www.indiacode.nic.in/",
}

MANIFEST_FIELDS = ("title", "issuer", "doc_id", "jurisdiction", "effective_date",
                   "version", "applicability", "doc_type", "url")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-download files already present")
    p.add_argument("--check", action="store_true", help="verify URLs without downloading")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def write_manifest(entry: dict, target: Path) -> None:
    data = {k: entry[k] for k in MANIFEST_FIELDS if k in entry}
    mpath = target.with_suffix(target.suffix + ".manifest.yaml")
    mpath.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    docs = registry.get("documents", [])

    ok = skipped = failed = 0
    for entry in docs:
        target = KB / entry["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        label = entry["file"]

        if target.exists() and not args.force and not args.check:
            write_manifest(entry, target)          # keep manifests in sync
            print(f"  have   {label}  ({target.stat().st_size:,} bytes)")
            skipped += 1
            continue

        try:
            from urllib.parse import urlparse

            headers = dict(HEADERS)
            host = urlparse(entry["url"]).netloc
            if host in REFERERS:
                headers["Referer"] = REFERERS[host]
            r = requests.get(entry["url"], headers=headers, timeout=args.timeout,
                             allow_redirects=True)
            r.raise_for_status()
            body = r.content
            ctype = r.headers.get("content-type", "").split(";")[0]

            # A 200 is not proof of a document: several of these hosts return an
            # HTML error page with a success code. Check the payload matches the
            # extension we are about to save it under.
            if target.suffix == ".pdf" and not body[:5].startswith(b"%PDF"):
                raise ValueError(f"not a PDF (content-type {ctype}, {len(body):,} bytes)")
            if len(body) < 5_000:
                raise ValueError(f"suspiciously small ({len(body):,} bytes) -- likely an error page")

            if args.check:
                print(f"  OK     {label}  {ctype} {len(body):,} bytes")
                ok += 1
                continue

            target.write_bytes(body)
            write_manifest(entry, target)
            print(f"  got    {label}  {ctype} {len(body):,} bytes")
            ok += 1
        except Exception as e:                     # noqa: BLE001
            print(f"  FAIL   {label}\n         {type(e).__name__}: {str(e)[:140]}")
            failed += 1

    print(f"\n{ok} fetched, {skipped} already present, {failed} failed")

    unavailable = registry.get("unavailable", [])
    if unavailable:
        print(f"\n{len(unavailable)} source(s) recorded as unavailable "
              f"(see the `unavailable:` block in sources.yaml):")
        for u in unavailable:
            print(f"  - {u['title']}")
            print(f"    {u['reason']}")

    if not args.check and (ok or skipped):
        print("\nNext: python -m notebooks.09_build_knowledge_base --dry-run")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
