#!/bin/sh
# Install Certbot post-renew hook so nginx in Docker picks up new PEMs.
# Run on the server once (requires root):
#   sudo bash deploy/install-certbot-renew-hook.sh
#
# Uses systemd certbot.timer (already enabled on Ubuntu) — no cron needed.

set -e
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SRC=$SCRIPT_DIR/certbot-renew-hook-embd-nginx.sh
DEST=/etc/letsencrypt/renewal-hooks/deploy/99-embd-nginx-reload.sh

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo bash $0" >&2
    exit 1
fi

if [ ! -f "$SRC" ]; then
    echo "Missing $SRC" >&2
    exit 1
fi

install -d /etc/letsencrypt/renewal-hooks/deploy
install -m 755 "$SRC" "$DEST"
chown root:root "$DEST"

echo "Installed $DEST"
echo "certbot.timer will run renew; this hook runs after a successful renew."
echo ""
echo "Configure the hook by setting environment variables:"
echo "  EMBD_COMPOSE_DIR=/path/to/embd"
echo "  EMBD_DOMAIN=embd.example.com"
echo ""
echo "Test nginx reload (no cert change):"
echo "  EMBD_COMPOSE_DIR=/path/to/embd EMBD_DOMAIN=embd.example.com \\"
echo "    RENEWED_LINEAGE=/etc/letsencrypt/live/embd.example.com $DEST"
