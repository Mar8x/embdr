#!/usr/bin/env bash
# Show stored contextual-generation throughput benchmarks from embd_meta.db.
# Usage: ./scripts/show-benchmarks.sh [path/to/chroma_db]

set -euo pipefail

DB_DIR="${1:-./chroma_db}"
DB="$DB_DIR/embd_meta.db"

if [[ ! -f "$DB" ]]; then
    echo "No metadata database found at: $DB"
    echo "Run 'embd ingest --recontext-file <file>' to measure throughput first."
    exit 1
fi

sqlite3 -column -header "$DB" \
    "SELECT model,
            machine_id,
            printf('%.0f', tok_per_sec) || ' tok/s' AS throughput,
            sample_chunks                            AS chunks_sampled,
            substr(measured_at, 1, 19)               AS measured_at
     FROM   ollama_benchmarks
     ORDER  BY measured_at DESC;"
