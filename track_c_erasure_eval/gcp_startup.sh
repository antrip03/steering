#!/bin/bash
# GCP Compute Engine startup script for a Track C erasure-eval run.
# Mirrors track_a_feature_discovery/gcp_startup.sh exactly in structure --
# see that file's comments for the full rationale (boot-time systemd
# service independent of any local connection, periodic + final log
# upload, set -e + trap for fail-loud/always-cleanup).
#
# Reads CONCEPTS (space-separated) / HF_TOKEN / GCS_BUCKET from instance
# metadata (set at instance creation, see gcp_launch.sh). Gemini calls go
# through Vertex AI using this VM's own default service account (ADC) --
# no API key metadata needed.
set -euo pipefail
exec > >(tee -a /var/log/pisces-run.log) 2>&1

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

CONCEPTS="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/concepts')"
HF_TOKEN="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token')"
GCS_BUCKET="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket')"
INSTANCE_NAME="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/name')"

log "=== Track C erasure eval starting: concepts=$CONCEPTS instance=$INSTANCE_NAME ==="

on_exit() {
  local exit_code=$?
  log "=== Script exiting with code $exit_code ==="
  kill "$LOG_UPLOADER_PID" 2>/dev/null || true
  # Copy any computed results straight to GCS, independent of --push-to-hub
  # (whose HF token currently lacks write access -- see run_erasure_eval.py's
  # own error output if hit). write_and_maybe_push() writes the local parquet
  # BEFORE attempting the hub push, so a concept's result survives here even
  # if run_erasure_eval.py goes on to crash on the push step.
  if compgen -G "/opt/pisces/artifacts/erasure_eval/*.parquet" > /dev/null; then
    gcloud storage cp /opt/pisces/artifacts/erasure_eval/*.parquet "gs://${GCS_BUCKET}/results/" || \
      log "WARNING: results copy to GCS failed"
  fi
  gcloud storage cp /var/log/pisces-run.log "gs://${GCS_BUCKET}/logs/${INSTANCE_NAME}.log" || \
    log "WARNING: final log upload failed -- log only survives via serial port output while the VM is still running"
  log "Shutting down to stop billing (disk persists until manually deleted)."
  shutdown -h now
}
trap on_exit EXIT

( while true; do
    sleep 120
    gcloud storage cp /var/log/pisces-run.log "gs://${GCS_BUCKET}/logs/${INSTANCE_NAME}.log" 2>/dev/null || true
  done ) &
LOG_UPLOADER_PID=$!

log "Installing Python 3.12..."
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

log "Pulling code + Track A results from GCS..."
mkdir -p /opt/pisces
cd /opt/pisces
# Track-C-specific tarball (includes artifacts/features/, Track A's
# completed per-concept output that load_selected_features() needs) --
# deliberately separate from Track A's own pisces-code.tar.gz, which
# excludes artifacts/ entirely since Track A generates that data rather
# than reading it.
gcloud storage cp "gs://${GCS_BUCKET}/pisces-code-track-c.tar.gz" .
tar xzf pisces-code-track-c.tar.gz

log "Setting up venv and installing dependencies..."
python3.12 -m venv /opt/pisces/venv
source /opt/pisces/venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

export HF_TOKEN
mkdir -p ~/.cache/huggingface
echo -n "$HF_TOKEN" > ~/.cache/huggingface/token

log "Starting run_erasure_eval.py..."
cd /opt/pisces/track_c_erasure_eval
CONCEPT_ARGS=()
IFS=',' read -ra CONCEPT_LIST <<< "$CONCEPTS"
for c in "${CONCEPT_LIST[@]}"; do
  CONCEPT_ARGS+=(--concept "$c")
done
python run_erasure_eval.py --device cuda "${CONCEPT_ARGS[@]}"

log "=== run_erasure_eval.py finished successfully ==="
