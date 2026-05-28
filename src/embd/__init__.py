"""embd — fully local document Q&A system optimized for Apple Silicon."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("embd")
except PackageNotFoundError:
    __version__ = "unknown"
