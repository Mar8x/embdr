#!/bin/sh
set -e

if [ -z "${EMBD_API_KEY}" ]; then
    echo "[entrypoint] ERROR: EMBD_API_KEY is empty. Set it in .env (see .env.example)." >&2
    exit 1
fi

# Public hostname for server_name and redirects (production: embd.example.com)
: "${EMBD_PUBLIC_HOST:=_}"
# Full https://host[:port] prefix for port-80 → HTTPS redirect (no trailing slash)
: "${EMBD_REDIRECT_TO_HTTPS:=https://127.0.0.1}"

CERT_DIR="/etc/nginx/certs"
: "${EMBD_TLS_FULLCHAIN:=${CERT_DIR}/fullchain.pem}"
: "${EMBD_TLS_PRIVKEY:=${CERT_DIR}/privkey.pem}"

have_cert=0
if [ -f "${EMBD_TLS_FULLCHAIN}" ] && [ -f "${EMBD_TLS_PRIVKEY}" ]; then
    have_cert=1
fi

if [ "$have_cert" -eq 0 ]; then
    if [ "${ALLOW_SELF_SIGNED:-0}" = "1" ]; then
        echo "[entrypoint] ALLOW_SELF_SIGNED=1 — generating a self-signed certificate (dev only)."
        openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
            -keyout "${CERT_DIR}/privkey.pem" \
            -out "${CERT_DIR}/fullchain.pem" \
            -subj "/CN=embd-docker-local"
        chmod 600 "${CERT_DIR}/privkey.pem"
        EMBD_TLS_FULLCHAIN="${CERT_DIR}/fullchain.pem"
        EMBD_TLS_PRIVKEY="${CERT_DIR}/privkey.pem"
        export EMBD_TLS_FULLCHAIN EMBD_TLS_PRIVKEY
    else
        echo "[entrypoint] ERROR: No TLS key material found." >&2
        echo "  Expected files: ${EMBD_TLS_FULLCHAIN} and ${EMBD_TLS_PRIVKEY}" >&2
        echo "  Production: Certbot (DNS-01); mount /etc/letsencrypt and set EMBD_TLS_* (see README)." >&2
        echo "  Dev only: set ALLOW_SELF_SIGNED=1 in the nginx service environment." >&2
        exit 1
    fi
fi

# Expand template — only these placeholders; nginx \$variables stay intact.
export EMBD_API_KEY EMBD_PUBLIC_HOST EMBD_REDIRECT_TO_HTTPS EMBD_TLS_FULLCHAIN EMBD_TLS_PRIVKEY
envsubst '${EMBD_API_KEY}${EMBD_PUBLIC_HOST}${EMBD_REDIRECT_TO_HTTPS}${EMBD_TLS_FULLCHAIN}${EMBD_TLS_PRIVKEY}' \
  < /etc/nginx/nginx.conf.template \
  > /etc/nginx/nginx.conf

# Refresh OpenAI IPs on startup
echo "[entrypoint] Updating OpenAI IP allowlist …"
/usr/local/bin/update-openai-ips.sh \
  || echo "[entrypoint] WARNING: OpenAI IP update failed, using existing list"

# Install daily cron job (03:00 UTC) — refresh IPs and reload nginx
echo "0 3 * * * /usr/local/bin/update-openai-ips.sh && nginx -s reload 2>&1 | logger -t openai-ips" \
  | crontab -

# Start cron in background
crond -l 2

# Validate config and start nginx in foreground
nginx -t
echo "[entrypoint] Starting nginx (server_name=${EMBD_PUBLIC_HOST}) …"
exec nginx -g 'daemon off;'
