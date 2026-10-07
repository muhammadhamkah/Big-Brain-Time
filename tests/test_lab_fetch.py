import io
import json
import re
import tempfile
import types
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from bigbrain import lab, net

HOUR = 3600 * 1000
DAY = 24 * HOUR


def ms(y, m, d, h=0):
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)


class FakeBinance:
    """Hourly candles from ``listed`` on, a REST API and a monthly archive. The candle still open is provisional: its
    close is 0.5 higher than the one it settles at. A month is archived a day after it ends."""

    def __init__(self, now, listed, market="perps", archive=True, missing=(), micros=False):
        self.now, self.listed, self.market, self.archive, self.missing, self.micros = now, listed, market, archive, set(missing), micros
        self.api_calls, self.archive_calls = 0, 0

    def row(self, t):
        close = 100.0 + (t // HOUR) % 97
        if t + HOUR > self.now:
            close += 0.5
        return [t, close, close + 1, close - 1, close, 10.0, t + HOUR - 1, 10.0 * close, 1, 0, 0, 0]

    def truth(self, start):
        first = max(self.listed, -(-start // HOUR) * HOUR)
        return [self.row(t) for t in range(first, self.now + 1, HOUR) if t <= self.now]

    def get(self, url, headers=None, **_):
        parts = urlsplit(url)
        if parts.hostname == "data.binance.vision":
            self.archive_calls += 1
            y, m = map(int, re.search(r"-(\d{4})-(\d{2})\.zip$", parts.path).groups())
            first = ms(y, m, 1)
            end = ms(y + (m == 12), m % 12 + 1, 1)
            if not self.archive or (y, m) in self.missing or end + DAY > self.now or end <= self.listed:
                raise net.HTTPStatusError(404, url, {}, b"")
            rows = [self.row(t) for t in range(max(first, self.listed), end, HOUR)]
            text = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n"
            text += "".join(",".join(str(v * 1000 if (i == 0 and self.micros) else v) for i, v in enumerate(r)) + "\n" for r in rows)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr(parts.path.rsplit("/", 1)[-1].replace(".zip", ".csv"), text)
            return buf.getvalue()
        self.api_calls += 1
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        start, limit = int(q["startTime"]), int(q["limit"])
        end = int(q.get("endTime", self.now))
        first = max(self.listed, -(-start // HOUR) * HOUR)
        rows = [self.row(t) for t in range(first, min(end, self.now) + 1, HOUR)][:limit]
        return json.dumps(rows).encode()


class FetchHistoryTests(unittest.TestCase):
    def run_fetch(self, fake, days, cache, market="perps"):
        clock = types.SimpleNamespace(time=lambda: fake.now / 1000, sleep=lambda s: None)
        with mock.patch.object(net, "http_get", side_effect=fake.get), mock.patch.object(lab, "time", clock):
            return lab.fetch_history("TESTUSDT", "1h", days, market, cache)

    def assertMatches(self, bars, fake, days):
        want = fake.truth(fake.now - days * DAY)
        self.assertEqual([b.date for b in bars], [lab._date(r[0]) for r in want])
        self.assertEqual([b.close for b in bars], [r[4] for r in want])

    def test_a_first_download_takes_whole_months_from_the_archive(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0)
        with tempfile.TemporaryDirectory() as cache:
            bars = self.run_fetch(fake, 400, cache)
        self.assertMatches(bars, fake, 400)
        self.assertEqual(fake.archive_calls, 13)  # Sep 2025 (trimmed) to Sep 2026
        self.assertLessEqual(fake.api_calls, 2)  # only October so far

    def test_a_later_run_fetches_only_what_is_new_and_settles_the_open_candle(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0)
        with tempfile.TemporaryDirectory() as cache:
            first = self.run_fetch(fake, 400, cache)
            self.assertEqual(first[-1].close, fake.row(fake.now - fake.now % HOUR)[4])
            fake.now += DAY + HOUR // 2
            fake.api_calls = fake.archive_calls = 0
            bars = self.run_fetch(fake, 400, cache)
            self.assertMatches(bars, fake, 400)
            self.assertEqual((fake.archive_calls, fake.api_calls), (0, 1))
            stale = [b for b in bars if b.date == first[-1].date][0]
            self.assertEqual(stale.close, first[-1].close - 0.5)  # was still open the first time

    def test_asking_further_back_fetches_only_the_older_part(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0)
        with tempfile.TemporaryDirectory() as cache:
            self.run_fetch(fake, 60, cache)
            fake.api_calls = fake.archive_calls = 0
            bars = self.run_fetch(fake, 200, cache)
        self.assertMatches(bars, fake, 200)
        self.assertLessEqual(fake.archive_calls, 6)
        self.assertLessEqual(fake.api_calls, 3)

    def test_old_per_day_files_seed_the_cache_and_are_removed(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0)
        with tempfile.TemporaryDirectory() as cache:
            old_now = ms(2026, 10, 6, 8)
            rows = [r for r in FakeBinance(now=old_now, listed=0).truth(old_now - 300 * DAY)]
            legacy = Path(cache) / "TESTUSDT-perps-1h-300d-20261006.json"
            legacy.write_text(json.dumps([{"date": lab._date(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4],
                                           "volume": r[5], "quote_volume": r[7]} for r in rows]))
            bars = self.run_fetch(fake, 300, cache)
            self.assertMatches(bars, fake, 300)
            self.assertEqual(fake.archive_calls, 0)
            self.assertLessEqual(fake.api_calls, 2)
            self.assertFalse(legacy.exists())
            self.assertTrue((Path(cache) / "TESTUSDT-perps-1h.json").exists())

    def test_months_missing_from_the_archive_come_from_the_api(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0, missing={(2026, 3), (2026, 4)})
        with tempfile.TemporaryDirectory() as cache:
            self.assertMatches(self.run_fetch(fake, 400, cache), fake, 400)
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0, archive=False)
        self.assertMatches(self.run_fetch(fake, 400, None), fake, 400)

    def test_a_late_listing_starts_at_the_listing_and_is_not_asked_again(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=ms(2026, 2, 14, 5))
        with tempfile.TemporaryDirectory() as cache:
            bars = self.run_fetch(fake, 400, cache)
            self.assertMatches(bars, fake, 400)
            self.assertEqual(bars[0].date, "2026-02-14 05:00")
            fake.api_calls = fake.archive_calls = 0
            self.run_fetch(fake, 400, cache)
            self.assertEqual((fake.archive_calls, fake.api_calls), (0, 1))

    def test_spot_archive_in_microseconds(self):
        fake = FakeBinance(now=ms(2026, 10, 7, 10), listed=0, market="spot", micros=True)
        with tempfile.TemporaryDirectory() as cache:
            self.assertMatches(self.run_fetch(fake, 120, cache, market="spot"), fake, 120)
        self.assertLessEqual(fake.api_calls, 2)  # the archive was read, not quietly replaced by the API


if __name__ == "__main__":
    unittest.main()
