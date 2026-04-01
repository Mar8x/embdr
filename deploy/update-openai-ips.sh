#!/bin/sh
# Fetch OpenAI's ChatGPT Actions CIDR list and write nginx geo directives.
#
# Run manually:   ./deploy/update-openai-ips.sh
# Runs in cron:   daily at 03:00 UTC inside the nginx container
#
# Output: deploy/openai-ips.conf (or /etc/nginx/conf.d/openai-ips.conf in container)
set -e

URL="https://openai.com/chatgpt-connectors.json"

# Detect output path: inside container vs host
if [ -f /etc/nginx/conf.d/openai-ips.conf ]; then
    OUTFILE="/etc/nginx/conf.d/openai-ips.conf"
elif [ -d "$(dirname "$0")" ]; then
    OUTFILE="$(cd "$(dirname "$0")" && pwd)/openai-ips.conf"
else
    OUTFILE="./deploy/openai-ips.conf"
fi

# Use a temp file under /tmp — bind-mounted OUTFILE cannot be replaced with mv
# (Docker gives "Resource busy"); cp over the mount target works.
TMPFILE="$(mktemp)"

echo "[update-openai-ips] Fetching ${URL} …"
JSON=$(curl -fsSL --max-time 30 "$URL")

if [ -z "$JSON" ]; then
    echo "[update-openai-ips] ERROR: empty response" >&2
    rm -f "$TMPFILE"
    exit 1
fi

# Parse JSON and write geo directives
echo "# Auto-generated from ${URL}" > "$TMPFILE"
echo "# Updated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$TMPFILE"
echo "#" >> "$TMPFILE"
echo "# DO NOT EDIT — this file is overwritten daily." >> "$TMPFILE"
echo "# Add your own IPs to custom-ips.conf instead." >> "$TMPFILE"
echo "" >> "$TMPFILE"

echo "$JSON" | jq -r '.prefixes[].ipv4Prefix' | while read -r cidr; do
    echo "${cidr} 1;"
done >> "$TMPFILE"

COUNT=$(grep -c ' 1;' "$TMPFILE" || true)
# Shell redirect overwrites bind-mounted files; busybox cp often fails with "File exists".
cat "$TMPFILE" > "$OUTFILE"
rm -f "$TMPFILE"

echo "[update-openai-ips] Wrote ${COUNT} CIDR entries to ${OUTFILE}"
