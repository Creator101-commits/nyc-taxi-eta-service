#!/usr/bin/env bash
# One-off: build the OSM graph dataset via the kaggle_graph kernel, then
# create the private dataset sreeharshak101/nyc-taxi-eta-osm-graph from its
# output. Run once before the first benchmark/smoke kernel.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

KAGGLE_BIN="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
META="kaggle_graph/kernel-metadata.json"
KERNEL_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META")"
SOURCES="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["dataset_sources"]))' "$META")"
SRC_DATASET="$(echo "$SOURCES" | awk '{print $1}')"
SRC_TITLE="$(basename "$SRC_DATASET")"
DATASET_ID="sreeharshak101/nyc-taxi-eta-osm-graph"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="artifacts/kaggle/${TS}-graph"

SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: working tree is dirty; the dataset reflects HEAD ($SHA) only" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/ds"
git archive HEAD | tar -x -C "$TMP/ds"
cat > "$TMP/ds/dataset-metadata.json" <<EOF
{"title": "${SRC_TITLE}", "id": "${SRC_DATASET}", "licenses": [{"name": "other"}]}
EOF
cat > "$TMP/ds/run_info.json" <<EOF
{"git_sha": "${SHA}", "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

if "$KAGGLE_BIN" datasets status "$SRC_DATASET" >/dev/null 2>&1; then
  "$KAGGLE_BIN" datasets version -p "$TMP/ds" -m "src ${SHA}" -r tar -q
else
  "$KAGGLE_BIN" datasets create -p "$TMP/ds" -r tar -q
fi

READY=0
for _ in $(seq 1 60); do
  STATUS="$("$KAGGLE_BIN" datasets status "$SRC_DATASET" 2>/dev/null || echo unknown)"
  echo "dataset $SRC_DATASET status: $STATUS"
  case "$STATUS" in
    *ready*|*complete*) READY=1; break ;;
  esac
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "error: src dataset not ready" >&2; exit 1; }
sleep 300

echo "pushing graph kernel $KERNEL_ID"
"$KAGGLE_BIN" kernels push -p "$(dirname "$META")"

TERMINAL=""
for _ in $(seq 1 360); do
  STATUS="$("$KAGGLE_BIN" kernels status "$KERNEL_ID" 2>&1 || echo "status-poll-error")"
  echo "kernel status: $STATUS"
  case "$STATUS" in
    *COMPLETE*|*ERROR*|*CANCEL*) TERMINAL="$STATUS"; break ;;
  esac
  sleep 60
done
case "$TERMINAL" in
  *COMPLETE*) ;;
  *) echo "error: graph kernel did not complete (${TERMINAL:-timeout})" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
"$KAGGLE_BIN" kernels output "$KERNEL_ID" -p "$OUT_DIR"

# The kernel writes graph_dataset/ under /kaggle/working; the download may
# nest it. Locate the payload dir.
PAYLOAD=""
for cand in "$OUT_DIR/graph_dataset" "$OUT_DIR"/*/graph_dataset; do
  if [ -f "$cand/nyc_drive_graph.graphml" ]; then PAYLOAD="$cand"; break; fi
done
if [ -z "$PAYLOAD" ]; then
  echo "error: graphml payload not found under $OUT_DIR" >&2
  find "$OUT_DIR" -maxdepth 3 -type f | head -20 >&2
  exit 1
fi

mkdir -p "$TMP/dsgraph"
cp "$PAYLOAD"/* "$TMP/dsgraph"/
cat > "$TMP/dsgraph/dataset-metadata.json" <<EOF
{"title": "nyc-taxi-eta-osm-graph", "id": "${DATASET_ID}", "licenses": [{"name": "other"}]}
EOF

if "$KAGGLE_BIN" datasets status "$DATASET_ID" >/dev/null 2>&1; then
  echo "versioning existing dataset $DATASET_ID"
  "$KAGGLE_BIN" datasets version -p "$TMP/dsgraph" -m "osm graph ${SHA}" -r zip -q
else
  echo "creating dataset $DATASET_ID"
  "$KAGGLE_BIN" datasets create -p "$TMP/dsgraph" -r zip -q
fi

echo "done: graph dataset=$DATASET_ID out=$OUT_DIR"
