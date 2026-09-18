import unittest
from unittest import mock

from bigbrain import net


class NetTests(unittest.TestCase):
    def _conn(self, status=200, body=b"<feed/>", headers=None, location=None):
        conn = mock.MagicMock()
        resp = conn.getresponse.return_value
        resp.status, resp.read.return_value = status, body
        resp.getheaders.return_value = list((headers or {}).items())
        resp.getheader.side_effect = lambda k: location if k == "Location" else None
        return conn

    def test_http_get_writes_curl_shaped_request(self):
        conn = self._conn()
        with mock.patch.object(net.http.client, "HTTPSConnection", return_value=conn), mock.patch.object(net.time, "sleep"):
            body = net.http_get("https://export.arxiv.org/api/query?search_query=all%3Ax")
        self.assertEqual(body, b"<feed/>")
        conn.putrequest.assert_called_once_with("GET", "/api/query?search_query=all%3Ax", skip_host=True, skip_accept_encoding=True)
        names = [c.args[0] for c in conn.putheader.call_args_list]
        self.assertEqual(names[:3], ["Host", "User-Agent", "Accept"])

    def test_http_get_follows_redirect_and_raises_on_error(self):
        first = self._conn(status=302, location="https://export.arxiv.org/x")
        second = self._conn(status=404, body=b"nope")
        with mock.patch.object(net.http.client, "HTTPSConnection", side_effect=[first, second]), mock.patch.object(net.time, "sleep"):
            with self.assertRaises(net.HTTPStatusError) as ctx:
                net.http_get("https://arxiv.org/x")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.body, b"nope")

    def test_http_redirect_is_upgraded_to_https(self):
        first = self._conn(status=301, location="http://feeds.example.com/x")
        second = self._conn(body=b"ok")
        with mock.patch.dict(net.os.environ, {"HTTPS_PROXY": "", "https_proxy": ""}, clear=False), \
             mock.patch.object(net.http.client, "HTTPSConnection", side_effect=[first, second]) as ctor, mock.patch.object(net.time, "sleep"):
            self.assertEqual(net.http_get("https://example.com/feed"), b"ok")
        self.assertEqual(ctor.call_args_list[1].args[0], "feeds.example.com")

    def test_http_json_and_gzip(self):
        import gzip, json
        payload = gzip.compress(json.dumps({"ok": 1}).encode())
        conn = self._conn(body=payload, headers={"Content-Encoding": "gzip"})
        with mock.patch.object(net.http.client, "HTTPSConnection", return_value=conn), mock.patch.object(net.time, "sleep"):
            self.assertEqual(net.http_json("https://api.github.com/x"), {"ok": 1})

    def test_proxy_resolution_honours_no_proxy(self):
        with mock.patch.dict(net.os.environ, {"HTTPS_PROXY": "http://127.0.0.1:3128", "NO_PROXY": "localhost,.internal.example"}, clear=False):
            self.assertEqual(net._proxy_for("www.reddit.com"), ("127.0.0.1", 3128))
            self.assertIsNone(net._proxy_for("api.internal.example"))
        with mock.patch.dict(net.os.environ, {"HTTPS_PROXY": "", "https_proxy": ""}, clear=False):
            self.assertIsNone(net._proxy_for("www.reddit.com"))

    def test_uses_tunnel_when_proxy_set(self):
        conn = self._conn()
        with mock.patch.dict(net.os.environ, {"HTTPS_PROXY": "http://127.0.0.1:3128", "NO_PROXY": ""}, clear=False), \
             mock.patch.object(net.http.client, "HTTPSConnection", return_value=conn) as ctor, mock.patch.object(net.time, "sleep"):
            net.http_get("https://www.reddit.com/r/x.json")
        self.assertEqual(ctor.call_args.args[:2], ("127.0.0.1", 3128))
        conn.set_tunnel.assert_called_once_with("www.reddit.com", 443)

    def test_rejects_plain_http(self):
        with self.assertRaises(ValueError):
            net.http_get("http://example.com")


if __name__ == "__main__":
    unittest.main()
