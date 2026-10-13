import pytest
from visual_servo_week6.frame_source import BrokerSource


@pytest.mark.parametrize(
    "url",
    [
        "file:///private",
        "https://127.0.0.1",
        "http://example.com",
        "http://localhost:8766/private",
        "http://name@localhost:8766",
    ],
)
def test_broker_accepts_only_local_http_origin_without_credentials(url):
    with pytest.raises(ValueError, match="local HTTP"):
        BrokerSource(url, (1280, 720))


def test_broker_does_not_follow_redirects_to_an_external_source(monkeypatch):
    import visual_servo_week6.frame_source

    class FakeConnection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ("127.0.0.1", 8766, 1)

        def request(self, method, path):
            assert (method, path) == ("GET", "/status")

        def getresponse(self):
            return type("Redirect", (), {"status": 302})()

        def close(self):
            pass

    monkeypatch.setattr(visual_servo_week6.frame_source, "HTTPConnection", FakeConnection)
    with pytest.raises(ValueError, match="redirects"):
        BrokerSource("http://localhost:8766", (1280, 720)).read()
