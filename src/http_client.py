import io
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RETRY_STATUSES = (429, 500, 502, 503, 504)


class HttpError(Exception):
    def __init__(self, url: str, status: int | None, message: str):
        super().__init__(f"{status or 'no response'} for {url}: {message}")
        self.url, self.status = url, status


class HttpClient:
    """Thin wrapper around a requests.Session bound to one API base URL."""

    def __init__(self, base_url: str, *, timeout: float = 30.0, retries: int = 3, backoff: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=RETRY_STATUSES,
            allowed_methods=("GET",),
            raise_on_status=False,  # return the last response so _get raises HttpError
        )
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session.mount("http://", HTTPAdapter(max_retries=retry))

    def get_json(self, path: str = "", params: dict | None = None):
        return self._get(path, params).json()

    def get_text(self, path: str = "", params: dict | None = None) -> str:
        return self._get(path, params).text

    def get_bytes(self, path: str = "", params: dict | None = None) -> bytes:
        return self._get(path, params).content

    def get_parquet(self, path: str = "", params: dict | None = None) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get_bytes(path, params)))

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, path: str, params: dict | None) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}" if path else self.base_url
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise HttpError(url, None, str(e)) from e
        if not resp.ok:
            raise HttpError(resp.url, resp.status_code, resp.text[:500])
        return resp
