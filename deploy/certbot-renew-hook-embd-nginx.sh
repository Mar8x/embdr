#!/bin/sh
# Certbot deploy hook: runs as root after a successful renewal (see certbot.timer).
# Installed under /etc/letsencrypt/renewal-hooks/deploy/ — use install-certbot-renew-hook.sh
#
# Env:
#   EMBD_COMPOSE_DIR — repo root with docker-compose.yml (required)
#   EMBD_DOMAIN      — your domain, e.g. embd.example.com (required)

COMPOSE_DIR=${EMBD_COMPOSE_DIR:?Set EMBD_COMPOSE_DIR to the repo root}
EMBD_DOMAIN=${EMBD_DOMAIN:?Set EMBD_DOMAIN to your domain}
DOCKER=/usr/bin/docker

# Only reload nginx when this certificate was renewed (avoids unrelated lineages).
if [ -n "${RENEWED_LINEAGE:-}" ]; then
    case "$RENEWED_LINEAGE" in
        */${EMBD_DOMAIN}) ;;
        *) exit 0 ;;
    esac
else
    # Manual test: set RENEWED_LINEAGE=/etc/letsencrypt/live/<your-domain>
    exit 0
fi

log() {
    logger -t certbot-embd "$*" 2>/dev/null || echo "[certbot-embd] $*"
}

if [ ! -d "$COMPOSE_DIR" ]; then
    log "skip: missing COMPOSE_DIR=$COMPOSE_DIR"
    exit 0
fi

# When COMPOSE_FILE is set (e.g. in .env), docker compose resolves the files
# from the env var — no explicit -f flags needed. This supports operator repos
# that layer additional compose files on top of the defaults.
if [ -n "${COMPOSE_FILE:-}" ]; then
    DC=("$DOCKER" compose --project-directory "$COMPOSE_DIR")
else
    DC=("$DOCKER" compose --project-directory "$COMPOSE_DIR"
        -f "$COMPOSE_DIR/docker-compose.yml"
        -f "$COMPOSE_DIR/deploy/docker-compose.le.yml")
fi

if ! "${DC[@]}" exec -T nginx nginx -s reload; then
    log "nginx -s reload failed, trying compose restart nginx"
    "${DC[@]}" restart nginx || log "nginx restart failed (check stack / docker)"
fi

log "nginx reload/restart attempted after renew for $EMBD_DOMAIN"
exit 0
