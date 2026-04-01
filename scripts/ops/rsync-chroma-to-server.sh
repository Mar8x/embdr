#!/usr/bin/env bash
# Sync local chroma_db/ → remote server.
#
#   REMOTE=user@host REMOTE_ROOT=/path/to/embd ./scripts/ops/rsync-chroma-to-server.sh
#   ./scripts/ops/rsync-chroma-to-server.sh --dry-run
#   ./scripts/ops/rsync-chroma-to-server.sh --stop-embd-first
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DB="${LOCAL_ROOT:-$ROOT}/chroma_db"
REMOTE="${REMOTE:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
DRY_RUN=()
STOP_FIRST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=(-n) ;;
    --stop-embd-first) STOP_FIRST=1 ;;
    -h|--help)
      echo "Usage: REMOTE=user@host REMOTE_ROOT=/path/to/embd $0 [--dry-run] [--stop-embd-first]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ -z "$REMOTE" || -z "$REMOTE_ROOT" ]]; then
  echo "Set REMOTE and REMOTE_ROOT, e.g.:" >&2
  echo "  REMOTE=user@host REMOTE_ROOT=/path/to/embd $0" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_DB" ]]; then
  echo "No local chroma_db at $LOCAL_DB" >&2
  exit 1
fi

DC_REMOTE='docker compose -f docker-compose.yml -f deploy/docker-compose.le.yml'

if [[ "$STOP_FIRST" -eq 1 && ${#DRY_RUN[@]} -eq 0 ]]; then
  echo "[rsync-chroma] Stopping embd on $REMOTE …"
  ssh "$REMOTE" "cd '$REMOTE_ROOT' && $DC_REMOTE stop embd"
elif [[ "$STOP_FIRST" -eq 1 ]]; then
  echo "[rsync-chroma] (dry-run: skipping remote stop/start)"
fi

echo "[rsync-chroma] rsync chroma_db/ → ${REMOTE}:${REMOTE_ROOT}/chroma_db/"
ssh "$REMOTE" "mkdir -p '${REMOTE_ROOT}/chroma_db'"
rsync -avz "${DRY_RUN[@]}" --progress --exclude '.DS_Store' -e ssh \
  "${LOCAL_DB}/" "${REMOTE}:${REMOTE_ROOT}/chroma_db/"

if [[ "$STOP_FIRST" -eq 1 && ${#DRY_RUN[@]} -eq 0 ]]; then
  echo "[rsync-chroma] Starting stack on $REMOTE …"
  ssh "$REMOTE" "cd '$REMOTE_ROOT' && $DC_REMOTE up -d"
fi

echo "[rsync-chroma] Done."
