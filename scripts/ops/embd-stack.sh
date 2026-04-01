#!/usr/bin/env bash
# Server-side Docker Compose helper (production LE overlay).
# Run on the host where the repo lives (e.g. after: cd ~/git/embd).
#
#   ./scripts/ops/embd-stack.sh start
#   ./scripts/ops/embd-stack.sh stop
#   ./scripts/ops/embd-stack.sh deploy          # git pull + build + up -d
#   ./scripts/ops/embd-stack.sh deploy-embd     # pull + rebuild embd only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${EMBD_ROOT:-$REPO_ROOT}"

DC=(docker compose -f docker-compose.yml -f deploy/docker-compose.le.yml)

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Commands:
  start, up           docker compose up -d
  stop                docker compose stop
  down                docker compose down
  restart [svc ...]   docker compose restart
  pull                git pull
  build [svc ...]     docker compose build
  ps, status          docker compose ps
  logs [svc]          docker compose logs -f --tail=100

  deploy              git pull && compose build && compose up -d
  deploy-embd         git pull && compose build embd && compose up -d --no-deps embd
  deploy-nginx        git pull && compose build nginx && compose up -d --no-deps nginx

Env:
  EMBD_ROOT   override repo directory (default: auto-detected from script path)
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  start|up)
    "${DC[@]}" up -d "$@"
    ;;
  stop)
    "${DC[@]}" stop "$@"
    ;;
  down)
    "${DC[@]}" down "$@"
    ;;
  restart)
    "${DC[@]}" restart "$@"
    ;;
  pull)
    git pull
    ;;
  build)
    "${DC[@]}" build "$@"
    ;;
  ps|status)
    "${DC[@]}" ps "$@"
    ;;
  logs)
    "${DC[@]}" logs -f --tail=100 "$@"
    ;;
  deploy)
    git pull
    "${DC[@]}" build
    "${DC[@]}" up -d
    ;;
  deploy-embd)
    git pull
    "${DC[@]}" build embd
    "${DC[@]}" up -d --no-deps embd
    ;;
  deploy-nginx)
    git pull
    "${DC[@]}" build nginx
    "${DC[@]}" up -d --no-deps nginx
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
