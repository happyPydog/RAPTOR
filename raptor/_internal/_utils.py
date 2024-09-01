"""Utilities function."""

import ssl
import urllib.request
import urllib.error


def ensure_ssl_verified(url: str) -> None:
    """
    Ensures SSL certificate is verified for the given URL. If verification fails,
    it falls back to an unverified SSL context.

    Args:
        url (str): The URL to verify SSL certificate for.
    """
    try:
        urllib.request.urlopen(url)
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        ssl._create_default_https_context = ssl._create_unverified_context
