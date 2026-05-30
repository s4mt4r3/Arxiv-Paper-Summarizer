"""
ArXiv scraper — direct API calls with exponential backoff.
Bypasses the arxiv-python library's pagination to handle 429/503 robustly.
"""
import fitz  # PyMuPDF
import requests
import xml.etree.ElementTree as ET
import json
import time
import tempfile
import os
import sys
from datetime import datetime, timedelta

OUTPUT_FILE = "arxiv_papers.json"
TARGET_COUNT = 5000
MAX_PDF_PAGES = 10
MAX_TEXT_CHARS = 2000
PAGE_SIZE = 100
PDF_DELAY = 2        # seconds between PDF downloads
API_DELAY = 30       # seconds between successful API page fetches

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

DATE_CUTOFF = datetime.utcnow() - timedelta(days=730)


def api_get(params: dict, max_retries: int = 8) -> requests.Response:
    """GET with exponential backoff on 429/503/network errors."""
    delay = 30
    for attempt in range(max_retries):
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            print(
                f"  [API {resp.status_code}] attempt {attempt+1}/{max_retries} "
                f"— sleeping {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 2, 600)
        except requests.RequestException as e:
            print(f"  [Network error] {e} — sleeping {delay}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 600)
    raise RuntimeError(f"ArXiv API failed after {max_retries} retries: {params}")


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", NS):
        # Published date
        pub_str = entry.findtext("atom:published", default="", namespaces=NS)
        try:
            pub_date = datetime.strptime(pub_str[:10], "%Y-%m-%d")
        except ValueError:
            continue

        if pub_date < DATE_CUTOFF:
            entries.append({"_too_old": True})
            continue

        # ID — strip version suffix
        entry_id = entry.findtext("atom:id", default="", namespaces=NS)
        paper_id = entry_id.split("/abs/")[-1]  # e.g. 2312.10997v1

        # PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        # Authors
        authors = [
            a.findtext("atom:name", default="", namespaces=NS)
            for a in entry.findall("atom:author", NS)
        ]

        entries.append(
            {
                "id": paper_id,
                "title": (entry.findtext("atom:title", default="", namespaces=NS) or "").strip(),
                "summary": (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip(),
                "url": pdf_url,
                "published_date": pub_str,
                "authors": authors,
            }
        )
    return entries


def fetch_pdf_text(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, timeout=30, stream=True)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            pages = min(MAX_PDF_PAGES, len(doc))
            text = "\n".join(doc[i].get_text() for i in range(pages))
            doc.close()
            return text[:MAX_TEXT_CHARS]
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"    [PDF error] {pdf_url}: {e}", file=sys.stderr)
        return ""


def load_existing() -> list:
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        with open(OUTPUT_FILE) as f:
            return json.load(f).get("papers", [])
    except Exception:
        return []


def save_papers(papers: list) -> None:
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"papers": papers}, f, indent=2, ensure_ascii=False)


def main():
    papers = load_existing()
    seen_ids = {p["id"] for p in papers}
    print(f"Resuming: {len(papers)} papers already collected.")

    offset = 0
    stop = False

    try:
        while len(papers) < TARGET_COUNT and not stop:
            params = {
                "search_query": "cat:cs.CL",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": offset,
                "max_results": PAGE_SIZE,
            }
            print(f"\n[API] Fetching offset={offset} ...")
            resp = api_get(params)
            entries = parse_entries(resp.text)

            if not entries:
                print("No more results from API.")
                break

            for entry in entries:
                if entry.get("_too_old"):
                    print("Reached papers older than 2 years. Stopping.")
                    stop = True
                    break

                if len(papers) >= TARGET_COUNT:
                    stop = True
                    break

                paper_id = entry["id"]
                if paper_id in seen_ids:
                    continue

                print(f"  [{len(papers)+1}/{TARGET_COUNT}] {paper_id} — {entry['title'][:55]}...")
                full_text = fetch_pdf_text(entry["url"]) if entry["url"] else ""

                papers.append({
                    "id": paper_id,
                    "title": entry["title"],
                    "full_text": full_text,
                    "summary": entry["summary"],
                    "url": entry["url"],
                    "published_date": entry["published_date"],
                    "authors": entry["authors"],
                })
                seen_ids.add(paper_id)

                if len(papers) % 100 == 0:
                    save_papers(papers)
                    print(f"  -> Checkpoint: {len(papers)} papers saved.")

                time.sleep(PDF_DELAY)

            offset += PAGE_SIZE
            if not stop:
                print(f"  [API delay] sleeping {API_DELAY}s before next page...")
                time.sleep(API_DELAY)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        save_papers(papers)
        print(f"\nDone. {len(papers)} papers saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
