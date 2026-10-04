"""End-to-end tests for the pwncorpus pipeline.

core.PREFIX_LEN is shrunk to 2 during tests so bucket membership is cheap to
control; every module reads it at call time, so server + core stay consistent.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pwncorpus import core, db, server, verifier

core.PREFIX_LEN = 2  # test-only: 256 buckets instead of 1M


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest().upper()


def pw_with_prefix(prefix: str, n: int, start: int = 0, tag: str = "pw"):
    """Deterministically find n distinct passwords whose SHA-1 starts with prefix."""
    out, i = [], start
    while len(out) < n:
        cand = f"{tag}-{i}"
        if sha1(cand).startswith(prefix):
            out.append(cand)
        i += 1
    return out


class CoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pwncorpus-test-")
        self.db_path = os.path.join(self.tmp, "corpus.db")
        db.init(self.db_path)
        self.conn = db.connect(self.db_path)
        self.addCleanup(self.conn.close)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    # -- helpers -----------------------------------------------------------
    def publish(self, k=2):
        info = core.build_candidate(self.conn, k)
        core.activate(self.conn, info["version"])
        return info["version"]

    def bucket(self, version, prefix):
        return self.conn.execute(
            "SELECT suffix, cnt FROM pub_buckets WHERE version=? AND prefix=?",
            (version, prefix)).fetchall()

    # -- tests -------------------------------------------------------------
    def test_ingest_dedup_same_batch_not_double_counted(self):
        items = ["alpha", "beta", "alpha", "gamma"]
        r1 = core.ingest_batch(self.conn, "breach-a", items)
        r2 = core.ingest_batch(self.conn, "breach-a", list(items))  # same content
        r3 = core.ingest_batch(self.conn, "breach-a", ["different", "set"],
                               idem_key="key-1")
        r4 = core.ingest_batch(self.conn, "breach-a", ["different", "set"],
                               idem_key="key-1")  # same idempotency key
        self.assertFalse(r1["duplicate"])
        self.assertTrue(r2["duplicate"])
        self.assertEqual(r1["batch_id"], r2["batch_id"])
        self.assertFalse(r3["duplicate"])
        self.assertTrue(r4["duplicate"])
        self.assertEqual(r3["batch_id"], r4["batch_id"])
        # occurrences of 'alpha' must be exactly 2 (from the single batch)
        version = self.publish(k=1)
        rows = self.conn.execute(
            "SELECT cnt FROM pub_buckets WHERE version=? AND suffix=?",
            (version, sha1("alpha")[core.PREFIX_LEN:])).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cnt"], 2)

    def test_concurrent_ingest_same_batch(self):
        items = [f"shared-{i}" for i in range(50)]
        results, errors = [], []

        def work():
            c = db.connect(self.db_path)
            try:
                results.append(core.ingest_batch(c, "breach-x", list(items)))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: work(), range(8)))
        self.assertEqual(errors, [])
        self.assertEqual(sum(1 for r in results if not r["duplicate"]), 1)
        self.assertEqual(len({r["batch_id"] for r in results}), 1)
        n = self.conn.execute("SELECT COUNT(*) c FROM batches").fetchone()["c"]
        self.assertEqual(n, 1)
        total = self.conn.execute(
            "SELECT COALESCE(SUM(cnt),0) s FROM batch_items").fetchone()["s"]
        self.assertEqual(total, 50)  # counted exactly once

    def test_concurrent_ingest_distinct_batches(self):
        def work(i):
            c = db.connect(self.db_path)
            try:
                return core.ingest_batch(c, f"src-{i}", [f"u-{i}-{j}" for j in range(20)])
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(work, range(8)))
        self.assertEqual(len({r["batch_id"] for r in results}), 8)
        total = self.conn.execute(
            "SELECT COALESCE(SUM(cnt),0) s FROM batch_items").fetchone()["s"]
        self.assertEqual(total, 160)

    def test_threshold_suppresses_small_buckets(self):
        lonely = pw_with_prefix("AB", 1, tag="lonely")          # bucket of 1
        trio = pw_with_prefix("CD", 3, tag="trio")              # bucket of 3
        core.ingest_batch(self.conn, "s", lonely + trio)
        version = self.publish(k=3)
        self.assertEqual(self.bucket(version, "AB"), [])        # suppressed
        self.assertEqual(len(self.bucket(version, "CD")), 3)    # kept
        v, body = core.query_range(self.conn, "AB")
        self.assertIsNone(body)                                 # not queryable
        v, body = core.query_range(self.conn, "CD")
        self.assertEqual(len(body.strip().split("\n")), 3)

    def test_retraction_only_via_new_publication(self):
        pw = pw_with_prefix("EF", 2, tag="victim")
        b1 = core.ingest_batch(self.conn, "leak-1", pw)
        v1 = self.publish(k=1)
        _, body = core.query_range(self.conn, "EF")
        self.assertIsNotNone(body)

        core.retract_batch(self.conn, b1["batch_id"])
        # current version untouched until a new one is built AND activated
        _, body = core.query_range(self.conn, "EF")
        self.assertIsNotNone(body)

        info = core.build_candidate(self.conn, k=1)
        _, body = core.query_range(self.conn, "EF")  # candidate not active yet
        self.assertIsNotNone(body)
        core.activate(self.conn, info["version"])
        _, body = core.query_range(self.conn, "EF")
        self.assertIsNone(body)  # gone from the serving version
        # old version still stored and exportable/verifiable
        out = os.path.join(self.tmp, "v1")
        core.export_version(self.conn, v1, out)
        self.assertEqual(verifier.verify_export(out), [])
        self.assertEqual(verifier.lookup(out, pw[0]), 1)

    def test_correction_removes_false_positive(self):
        good = pw_with_prefix("11", 2, tag="good")
        bad = pw_with_prefix("22", 1, tag="oops")
        b = core.ingest_batch(self.conn, "leak-2", good + bad)
        core.add_corrections(self.conn, b["batch_id"], [bad[0]])
        self.publish(k=1)
        _, body = core.query_range(self.conn, "22")
        self.assertIsNone(body)
        _, body = core.query_range(self.conn, "11")
        self.assertEqual(len(body.strip().split("\n")), 2)

    def test_threshold_change_requires_new_version_and_old_stays_valid(self):
        pair = pw_with_prefix("33", 2, tag="pair")
        core.ingest_batch(self.conn, "s", pair)
        v1 = self.publish(k=2)
        self.assertEqual(len(self.bucket(v1, "33")), 2)
        v2 = self.publish(k=3)  # stricter threshold: bucket disappears
        self.assertEqual(self.bucket(v2, "33"), [])
        _, body = core.query_range(self.conn, "33")
        self.assertIsNone(body)
        # v1 export still verifies against ITS OWN threshold
        out1, out2 = os.path.join(self.tmp, "v1"), os.path.join(self.tmp, "v2")
        core.export_version(self.conn, v1, out1)
        core.export_version(self.conn, v2, out2)
        self.assertEqual(verifier.verify_export(out1), [])
        self.assertEqual(verifier.verify_export(out2), [])
        with open(os.path.join(out1, "manifest.json")) as fh:
            self.assertEqual(json.load(fh)["threshold_k"], 2)

    def test_export_tampering_detected(self):
        core.ingest_batch(self.conn, "s", pw_with_prefix("44", 3, tag="t"))
        v = self.publish(k=2)
        out = os.path.join(self.tmp, "exp")
        core.export_version(self.conn, v, out)
        self.assertEqual(verifier.verify_export(out), [])
        # tamper: inflate a count
        f = os.path.join(out, "buckets", "44.txt")
        lines = open(f).read().splitlines()
        s, c = lines[0].split(":")
        lines[0] = f"{s}:{int(c) + 100}"
        open(f, "w").write("\n".join(lines) + "\n")
        self.assertTrue(any("sha256" in e for e in verifier.verify_export(out)))

    def test_verifier_rejects_below_threshold_bucket(self):
        core.ingest_batch(self.conn, "s", pw_with_prefix("55", 3, tag="z"))
        v = self.publish(k=2)
        out = os.path.join(self.tmp, "exp")
        core.export_version(self.conn, v, out)
        # hand-craft a below-threshold bucket + matching manifest entry
        body = "0" * (40 - core.PREFIX_LEN) + ":1\n"
        open(os.path.join(out, "buckets", "99.txt"), "w").write(body)
        mpath = os.path.join(out, "manifest.json")
        m = json.load(open(mpath))
        m["files"]["99"] = hashlib.sha256(body.encode()).hexdigest()
        payload = {f: m[f] for f in
                   ("version", "hash_algo", "prefix_len", "threshold_k",
                    "created_at", "files")}
        m["manifest_sha256"] = hashlib.sha256(
            core.canonical_json(payload).encode()).hexdigest()
        json.dump(m, open(mpath, "w"))
        errors = verifier.verify_export(out)
        self.assertTrue(any("below threshold" in e for e in errors), errors)

    def test_no_raw_password_in_storage(self):
        secret = "correct-horse-battery-staple-!9"
        core.ingest_batch(self.conn, "leak", [secret, "other-1", "other-2"])
        self.publish(k=1)
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()
        with open(self.db_path, "rb") as fh:
            raw = fh.read()
        self.assertNotIn(secret.encode(), raw)
        self.assertIn(sha1(secret).encode(), raw)  # fingerprint IS there
        self.conn = db.connect(self.db_path)

    def test_stale_candidate_rejected_then_refresh_keeps_threshold(self):
        core.ingest_batch(self.conn, "s1", pw_with_prefix("A0", 2, tag="one"))
        v1 = self.publish(k=2)
        info = core.build_candidate(self.conn, k=2)
        v2 = info["version"]
        # no source change yet: draft is fresh and review agrees
        state = core.review_state(self.conn)
        self.assertFalse(state["candidate"]["stale"])
        self.assertEqual(state["corpus_sha256"], info["corpus_sha256"])
        # source data changes AFTER the draft was built
        core.ingest_batch(self.conn, "s2", pw_with_prefix("A1", 2, tag="two"))
        state = core.review_state(self.conn)
        self.assertTrue(state["candidate"]["stale"])
        # activation must not let the old snapshot overwrite the newer state
        with self.assertRaises(core.Conflict):
            core.activate(self.conn, v2)
        self.assertEqual(core.current_version(self.conn), v1)  # serving unchanged
        _, body = core.query_range(self.conn, "A1")
        self.assertIsNone(body)  # new batch not served either
        # refresh keeps the original threshold and unblocks activation
        r = core.refresh_candidate(self.conn, v2)
        self.assertEqual(r["threshold_k"], 2)
        self.assertTrue(r["was_stale"])
        self.assertFalse(core.review_state(self.conn)["candidate"]["stale"])
        core.activate(self.conn, v2)
        self.assertEqual(core.current_version(self.conn), v2)
        _, body = core.query_range(self.conn, "A1")
        self.assertEqual(len(body.strip().split("\n")), 2)

    def test_retraction_and_correction_make_candidate_stale(self):
        pw = pw_with_prefix("B0", 2, tag="victim")
        b = core.ingest_batch(self.conn, "leak", pw)
        self.publish(k=1)
        info = core.build_candidate(self.conn, k=1)
        core.retract_batch(self.conn, b["batch_id"])
        self.assertTrue(core.review_state(self.conn)["candidate"]["stale"])
        with self.assertRaises(core.Conflict):
            core.activate(self.conn, info["version"])
        # refresh picks up the retraction; activation then works
        core.refresh_candidate(self.conn, info["version"])
        core.activate(self.conn, info["version"])
        _, body = core.query_range(self.conn, "B0")
        self.assertIsNone(body)
        # same for corrections on the next round
        pw2 = pw_with_prefix("B1", 2, tag="oops")
        b2 = core.ingest_batch(self.conn, "leak-2", pw2)
        info2 = core.build_candidate(self.conn, k=1)
        core.add_corrections(self.conn, b2["batch_id"], [pw2[0]])
        with self.assertRaises(core.Conflict):
            core.activate(self.conn, info2["version"])
        core.refresh_candidate(self.conn, info2["version"])
        core.activate(self.conn, info2["version"])
        _, body = core.query_range(self.conn, "B1")
        self.assertEqual(len(body.strip().split("\n")), 1)  # corrected item gone

    def test_activate_vs_source_update_ordering(self):
        core.ingest_batch(self.conn, "s1", pw_with_prefix("C0", 2, tag="c1"))
        self.publish(k=2)
        # activation commits first: the later ingest waits for the next version
        info = core.build_candidate(self.conn, k=2)
        core.activate(self.conn, info["version"])
        core.ingest_batch(self.conn, "s2", pw_with_prefix("C1", 2, tag="c2"))
        _, body = core.query_range(self.conn, "C1")
        self.assertIsNone(body)  # not in the serving version
        # source update commits first: confirmation has no effect
        info2 = core.build_candidate(self.conn, k=2)
        core.ingest_batch(self.conn, "s3", pw_with_prefix("C2", 2, tag="c3"))
        with self.assertRaises(core.Conflict):
            core.activate(self.conn, info2["version"])
        self.assertEqual(core.current_version(self.conn), info["version"])

    def test_concurrent_activate_and_ingest_resolve_to_one_order(self):
        core.ingest_batch(self.conn, "s1", pw_with_prefix("D0", 2, tag="d1"))
        v1 = self.publish(k=2)
        info = core.build_candidate(self.conn, k=2)
        results = {}

        def do_ingest():
            c = db.connect(self.db_path)
            try:
                core.ingest_batch(c, "s2", pw_with_prefix("D1", 2, tag="d2"))
                results["ingest"] = "ok"
            finally:
                c.close()

        def do_activate():
            c = db.connect(self.db_path)
            try:
                core.activate(c, info["version"])
                results["activate"] = "ok"
            except core.Conflict:
                results["activate"] = "conflict"
            finally:
                c.close()

        threads = [threading.Thread(target=do_ingest),
                   threading.Thread(target=do_activate)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results["ingest"], "ok")
        if results["activate"] == "ok":
            # activation committed first: candidate serves, racing batch is
            # left for the next version
            self.assertEqual(core.current_version(self.conn), info["version"])
            _, body = core.query_range(self.conn, "D1")
            self.assertIsNone(body)
        else:
            # ingest committed first: confirmation rejected, serving unchanged
            self.assertEqual(core.current_version(self.conn), v1)

    def test_refresh_rejected_for_non_candidate(self):
        core.ingest_batch(self.conn, "s", pw_with_prefix("E0", 2, tag="e"))
        v1 = self.publish(k=2)  # active, not candidate
        with self.assertRaises(core.Conflict):
            core.refresh_candidate(self.conn, v1)
        with self.assertRaises(core.NotFound):
            core.refresh_candidate(self.conn, 999)


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pwncorpus-http-")
        self.db_path = os.path.join(self.tmp, "corpus.db")
        db.init(self.db_path)
        conn = db.connect(self.db_path)
        self.tok_ingest = core.create_token(conn, "ing", ["ingest"])
        self.tok_pub = core.create_token(conn, "pub", ["publish"])
        self.tok_retract = core.create_token(conn, "ret", ["retract"])
        conn.close()
        self.srv = server.make_server(self.db_path, "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.srv.shutdown)
        self.addCleanup(self.srv.server_close)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def http(self, method, path, body=None, token=None, raw=False):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as resp:
                payload = resp.read()
                return resp.status, (payload if raw else json.loads(payload)), dict(resp.headers)
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload), dict(e.headers)
            except json.JSONDecodeError:
                return e.code, payload, dict(e.headers)

    def publish(self, k=2):
        st, body, _ = self.http("POST", "/api/publications/build", {"k": k},
                                token=self.tok_pub)
        self.assertEqual(st, 201, body)
        v = body["version"]
        st, body, _ = self.http("POST", f"/api/publications/{v}/activate",
                                token=self.tok_pub)
        self.assertEqual(st, 200, body)
        return v

    def test_authz_enforced(self):
        st, _, _ = self.http("POST", "/api/batches",
                             {"source": "x", "items": ["a"]})
        self.assertEqual(st, 401)
        st, _, _ = self.http("POST", "/api/batches",
                             {"source": "x", "items": ["a"]}, token=self.tok_pub)
        self.assertEqual(st, 403)  # publish token may not ingest
        st, _, _ = self.http("POST", "/api/publications/build", {"k": 1},
                             token=self.tok_ingest)
        self.assertEqual(st, 403)  # ingest token may not publish
        st, body, _ = self.http("POST", "/api/batches",
                                {"source": "x", "items": ["a"]},
                                token=self.tok_ingest)
        self.assertEqual(st, 201, body)
        bid = body["batch_id"]
        st, _, _ = self.http("POST", f"/api/batches/{bid}/retract",
                             token=self.tok_ingest)
        self.assertEqual(st, 403)  # ingest token may not retract
        st, _, _ = self.http("POST", f"/api/batches/{bid}/retract",
                             token=self.tok_retract)
        self.assertEqual(st, 200)

    def test_range_api_and_version_header(self):
        pws = pw_with_prefix("77", 3, tag="api")
        self.http("POST", "/api/batches", {"source": "s", "items": pws},
                  token=self.tok_ingest)
        st, body, _ = self.http("GET", "/api/range/77")
        self.assertEqual(st, 503)  # nothing published yet
        v = self.publish(k=2)
        st, body, hdr = self.http("GET", "/api/range/77", raw=True)
        self.assertEqual(st, 200)
        self.assertEqual(hdr.get("X-Corpus-Version"), str(v))
        lines = body.decode().strip().split("\n")
        self.assertEqual(len(lines), 3)
        suffixes = {l.split(":")[0] for l in lines}
        self.assertEqual(suffixes, {sha1(p)[core.PREFIX_LEN:] for p in pws})
        st, body, _ = self.http("GET", "/api/range/zz")
        self.assertEqual(st, 400)  # invalid prefix
        st, body, _ = self.http("GET", "/api/range/99")
        self.assertEqual(st, 404)  # unknown bucket

    def test_no_raw_password_in_api_surface(self):
        secret = "sup3r-secret-p@ssphrase"
        st, body, _ = self.http("POST", "/api/batches",
                                {"source": "s", "items": [secret, "x", "x"]},
                                token=self.tok_ingest)
        self.assertEqual(st, 201)
        self.assertNotIn(secret, json.dumps(body))
        self.publish(k=1)
        prefix = sha1(secret)[:core.PREFIX_LEN]
        st, body, hdr = self.http("GET", f"/api/range/{prefix}", raw=True)
        self.assertNotIn(secret.encode(), body)
        st, body, _ = self.http("GET", "/api/publications", token=self.tok_pub)
        self.assertNotIn(secret, json.dumps(body))
        st, body, _ = self.http("GET", f"/review?token={self.tok_pub}", raw=True)
        self.assertEqual(st, 200)
        self.assertNotIn(secret.encode(), body)

    def test_duplicate_batch_over_http(self):
        payload = {"source": "s", "items": ["a", "b", "a"], "idempotency_key": "k1"}
        st1, b1, _ = self.http("POST", "/api/batches", payload, token=self.tok_ingest)
        st2, b2, _ = self.http("POST", "/api/batches", payload, token=self.tok_ingest)
        self.assertEqual((st1, st2), (201, 200))
        self.assertEqual(b1["batch_id"], b2["batch_id"])
        self.assertTrue(b2["duplicate"])

    def test_no_version_mixing_during_activation(self):
        # v1: bucket 80 has suffixes from batch A; v2 adds batch B suffixes.
        pa = pw_with_prefix("80", 2, tag="batchA")
        pb = pw_with_prefix("80", 2, tag="batchB")
        self.http("POST", "/api/batches", {"source": "A", "items": pa},
                  token=self.tok_ingest)
        v1 = self.publish(k=2)
        self.http("POST", "/api/batches", {"source": "B", "items": pb},
                  token=self.tok_ingest)
        st, cand, _ = self.http("POST", "/api/publications/build", {"k": 2},
                                token=self.tok_pub)
        v2 = cand["version"]

        _, body_v1, _ = self.http("GET", "/api/range/80", raw=True)
        expected = {}
        conn = db.connect(self.db_path)
        for v in (v1,):
            _, b = core.query_range(conn, "80")
            expected[v] = b
        conn.close()

        stop = threading.Event()
        violations = []

        def reader():
            while not stop.is_set():
                try:
                    st, body, hdr = self.http("GET", "/api/range/80", raw=True)
                except Exception:
                    continue
                if st != 200:
                    continue
                ver = int(hdr["X-Corpus-Version"])
                text = body.decode()
                ok = (ver == v1 and text == expected[v1]) or \
                     (ver == v2 and len(text.strip().split("\n")) == 4
                      and ver == int(hdr["X-Corpus-Version"]))
                if not ok:
                    violations.append((ver, text))

        def activator():
            import time
            time.sleep(0.15)
            self.http("POST", f"/api/publications/{v2}/activate",
                      token=self.tok_pub)

        readers = [threading.Thread(target=reader) for _ in range(6)]
        act = threading.Thread(target=activator)
        for t in readers:
            t.start()
        act.start()
        act.join()
        import time
        time.sleep(0.4)
        stop.set()
        for t in readers:
            t.join()
        # every response must be a complete, single-version body
        self.assertEqual(violations, [])
        # and after the switch, only v2 is served
        _, body, hdr = self.http("GET", "/api/range/80", raw=True)
        self.assertEqual(int(hdr["X-Corpus-Version"]), v2)
        self.assertEqual(len(body.decode().strip().split("\n")), 4)

    def test_retired_version_not_served_but_verifiable(self):
        pws = pw_with_prefix("90", 2, tag="old")
        st, b, _ = self.http("POST", "/api/batches",
                             {"source": "s", "items": pws}, token=self.tok_ingest)
        bid = b["batch_id"]
        v1 = self.publish(k=2)
        # retract + rebuild + activate
        self.http("POST", f"/api/batches/{bid}/retract", token=self.tok_retract)
        v2 = self.publish(k=2)
        st, body, hdr = self.http("GET", "/api/range/90")
        self.assertEqual((st, hdr.get("X-Corpus-Version")), (404, str(v2)))
        # v1 manifest still fetchable (verifiability), export verifies
        st, manifest, _ = self.http("GET", f"/api/publications/{v1}/manifest")
        self.assertEqual(st, 200)
        self.assertEqual(manifest["status"], "retired")
        conn = db.connect(self.db_path)
        out = os.path.join(self.tmp, "v1-export")
        core.export_version(conn, v1, out)
        conn.close()
        self.assertEqual(verifier.verify_export(out), [])
        self.assertEqual(verifier.lookup(out, pws[0]), 1)

    def test_review_page_requires_publish_scope(self):
        st, _, _ = self.http("GET", "/review")
        self.assertEqual(st, 401)
        st, _, _ = self.http("GET", f"/review?token={self.tok_ingest}")
        self.assertEqual(st, 403)
        st, body, _ = self.http("GET", f"/review?token={self.tok_pub}", raw=True)
        self.assertEqual(st, 200)
        self.assertIn(b"Publication review", body)

    def test_stale_candidate_flow_over_http(self):
        self.http("POST", "/api/batches",
                  {"source": "s", "items": pw_with_prefix("A0", 2, tag="h1")},
                  token=self.tok_ingest)
        v1 = self.publish(k=2)
        st, cand, _ = self.http("POST", "/api/publications/build", {"k": 2},
                                token=self.tok_pub)
        v2 = cand["version"]
        # fresh draft: detail says not stale, review page offers activation
        st, det, _ = self.http("GET", f"/api/publications/{v2}", token=self.tok_pub)
        self.assertFalse(det["stale"])
        st, page, _ = self.http("GET", f"/review?token={self.tok_pub}", raw=True)
        self.assertIn(b"Activate v", page)
        self.assertNotIn(b"STALE", page)
        # source data changes after the draft was built
        self.http("POST", "/api/batches",
                  {"source": "s2", "items": pw_with_prefix("A1", 2, tag="h2")},
                  token=self.tok_ingest)
        st, det, _ = self.http("GET", f"/api/publications/{v2}", token=self.tok_pub)
        self.assertTrue(det["stale"])
        self.assertNotEqual(det["current_corpus_sha256"],
                            det["publication"]["corpus_sha256"])
        # review page flags it and offers refresh instead of activation
        st, page, _ = self.http("GET", f"/review?token={self.tok_pub}", raw=True)
        self.assertIn(b"STALE", page)
        self.assertIn(b"/review/refresh", page)
        # activation is rejected; the serving version is untouched
        st, body, _ = self.http("POST", f"/api/publications/{v2}/activate",
                                token=self.tok_pub)
        self.assertEqual(st, 409)
        self.assertIn("stale", body["error"])
        _, _, hdr = self.http("GET", "/api/range/A0", raw=True)
        self.assertEqual(hdr.get("X-Corpus-Version"), str(v1))
        # refresh keeps the threshold and unblocks activation
        st, body, _ = self.http("POST", f"/api/publications/{v2}/refresh",
                                token=self.tok_pub)
        self.assertEqual(st, 200)
        self.assertEqual(body["threshold_k"], 2)
        self.assertTrue(body["was_stale"])
        st, det, _ = self.http("GET", f"/api/publications/{v2}", token=self.tok_pub)
        self.assertFalse(det["stale"])
        st, body, _ = self.http("POST", f"/api/publications/{v2}/activate",
                                token=self.tok_pub)
        self.assertEqual(st, 200)
        st, body, hdr = self.http("GET", "/api/range/A1", raw=True)
        self.assertEqual(st, 200)
        self.assertEqual(hdr.get("X-Corpus-Version"), str(v2))
        # an active version can no longer be refreshed
        st, _, _ = self.http("POST", f"/api/publications/{v2}/refresh",
                             token=self.tok_pub)
        self.assertEqual(st, 409)


if __name__ == "__main__":
    unittest.main()
