"""LLM token accounting returned by generators (prompt / completion / total)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one model call (values may be partial if the backend omits fields)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None


def add_usage(session_prompt: int, session_completion: int, usage: TokenUsage | None) -> tuple[int, int]:
    """Return updated running totals; only adds fields that are not None."""
    if usage is None:
        return session_prompt, session_completion
    p, c = session_prompt, session_completion
    if usage.prompt_tokens is not None:
        p += usage.prompt_tokens
    if usage.completion_tokens is not None:
        c += usage.completion_tokens
    return p, c


def format_usage_plain(
    usage: TokenUsage | None,
    *,
    embed_ms: float,
    retrieve_ms: float,
    generate_s: float,
    turn_total_s: float,
    session_prompt: int,
    session_completion: int,
) -> str:
    """Plain-text (non-markdown) stats block for terminal output."""
    lines: list[str] = []

    lines.append("  ── Timing ─────────────────────────────────────────")
    lines.append(f"  {'Embed:':<16} {embed_ms:>10.1f} ms")
    lines.append(f"  {'Retrieve:':<16} {retrieve_ms:>10.1f} ms")
    lines.append(f"  {'Generate:':<16} {generate_s:>10.2f} s")
    lines.append(f"  {'Full turn:':<16} {turn_total_s:>10.2f} s")

    lines.append("  ── LLM tokens (this question) ─────────────────────")
    if usage is None:
        lines.append("  (no model call — nothing sent to the LLM)")
    else:
        pi = f"{usage.prompt_tokens:,}" if usage.prompt_tokens is not None else "—"
        co = f"{usage.completion_tokens:,}" if usage.completion_tokens is not None else "—"
        tot = f"{usage.total_tokens:,}" if usage.total_tokens is not None else "—"
        lines.append(f"  {'Tokens in:':<16} {pi:>10}")
        lines.append(f"  {'Tokens out:':<16} {co:>10}")
        lines.append(f"  {'Tokens total:':<16} {tot:>10}")

    st = session_prompt + session_completion
    lines.append("  ── LLM tokens (session, since shell start) ───────")
    lines.append(f"  {'Tokens in:':<16} {session_prompt:>10,}")
    lines.append(f"  {'Tokens out:':<16} {session_completion:>10,}")
    lines.append(f"  {'Tokens total:':<16} {st:>10,}")

    return "\n".join(lines) + "\n"
