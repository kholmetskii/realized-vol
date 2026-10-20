"""Errors raised while downloading Dukascopy data."""


class FetchError(RuntimeError):
    """One hour could not be downloaded, after every attempt."""
