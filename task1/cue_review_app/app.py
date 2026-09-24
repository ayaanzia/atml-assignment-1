#!/usr/bin/env python3
"""Serve the blind cue-conflict review GUI on localhost."""
from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from workflow import (
    balanced_approved_rows,
    read_candidates,
    read_ratings,
    remove_rating,
    review_summary,
    save_rating,
    write_csv_atomic,
)


APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, handler, manifest: Path, ratings: Path, output: Path, quota: int):
        super().__init__(address, handler)
        self.manifest = manifest.resolve()
        self.ratings = ratings.resolve()
        self.output = output.resolve()
        self.quota = quota
        self.candidates = read_candidates(self.manifest)
        self.by_id = {row["cue_id"]: row for row in self.candidates}


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def _json(self, value, status=HTTPStatus.OK):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            return self._file(APP_DIR / "index.html", "text/html; charset=utf-8")
        if route == "/api/state":
            ratings = read_ratings(self.server.ratings)
            return self._json({
                "candidates": self.server.candidates,
                "ratings": ratings,
                "summary": review_summary(self.server.candidates, ratings),
                "quota": self.server.quota,
                "output": str(self.server.output),
            })
        if route.startswith("/images/"):
            cue_id = unquote(route.removeprefix("/images/"))
            row = self.server.by_id.get(cue_id)
            if row is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            image = (self.server.manifest.parent / row["image_path"]).resolve()
            if self.server.manifest.parent not in image.parents:
                return self.send_error(HTTPStatus.FORBIDDEN)
            return self._file(image, mimetypes.guess_type(image)[0] or "application/octet-stream")
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            data = self._body()
            if route == "/api/rate":
                cue_id = data["cue_id"]
                if cue_id not in self.server.by_id:
                    raise ValueError("Unknown cue_id")
                ratings = save_rating(
                    self.server.ratings, cue_id, data["rating"],
                    datetime.now(timezone.utc).isoformat(),
                )
                return self._json({"summary": review_summary(self.server.candidates, ratings)})
            if route == "/api/undo":
                ratings = remove_rating(self.server.ratings, data["cue_id"])
                return self._json({"summary": review_summary(self.server.candidates, ratings)})
            if route == "/api/export":
                ratings = read_ratings(self.server.ratings)
                rows = balanced_approved_rows(
                    self.server.candidates, ratings, quota=self.server.quota, seed=6304
                )
                write_csv_atomic(self.server.output, rows)
                return self._json({"ok": True, "rows": len(rows), "path": str(self.server.output)})
            self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _file(self, path: Path, content_type: str):
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TASK_DIR / "results/cue_conflict_candidates/candidates.csv")
    parser.add_argument("--ratings", type=Path, default=TASK_DIR / "results/cue_conflict_candidates/review_ratings.json")
    parser.add_argument("--output", type=Path, default=TASK_DIR / "results/cue_conflict_approved_manifest.csv")
    parser.add_argument("--quota", type=int, default=20)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ReviewServer(("127.0.0.1", args.port), Handler, args.manifest, args.ratings, args.output, args.quota)
    print(f"Review GUI: http://127.0.0.1:{args.port}")
    print(f"Decisions:  {server.ratings}")
    print(f"Selection:  {server.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
