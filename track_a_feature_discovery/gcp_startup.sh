#!/bin/bash
# GCP Compute Engine startup script for a Track A production run.
#
# Runs as a systemd service (google-startup-scripts.service) triggered by the
# VM's own boot process -- entirely independent of any SSH session or local
# machine's network/power state. If the machine that launched this VM loses
# network or gets shut down, this keeps running exactly as if nothing
# happened, the same guarantee Modal's --detach gives on that platform.
#
# Reads CONCEPT / LAYERS / HF_TOKEN from instance metadata (set at instance
# creation, see gcp_launch.sh) rather than being baked into this script, so
# the same script serves every concept without editing.
#
# Logging: everything below is teed to /var/log/pisces-run.log AND to
# stdout/stderr, which google-startup-scripts.service captures into the
# instance's serial port 1 output (retrievable with zero SSH access via
# `gcloud compute instances get-serial-port-output`) and, on Deep Learning VM
# images, into Cloud Logging via the pre-installed ops agent. The full log
# file is also copied to GCS at the end (and periodically during the run)
# so it survives even if the VM is later deleted.
set -euo pipefail
exec > >(tee -a /var/log/pisces-run.log) 2>&1

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

CONCEPT="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/concept')"
LAYERS="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/layers')"
HF_TOKEN="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token')"
GCS_BUCKET="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket')"
INSTANCE_NAME="$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/name')"

log "=== Track A production run starting: concept=$CONCEPT layers=$LAYERS instance=$INSTANCE_NAME ==="

# set -e means any failed command below exits the script immediately rather
# than cascading silently into later steps operating on missing/partial
# state -- a real run hit exactly that: the code-tarball download failed
# (IAM permissions, since fixed), but without set -e the script just kept
# going, cd'd into a directory that was never created, and "python
# discover.py" ran from the wrong cwd with a confusing "file not found"
# instead of the actual, immediately-obvious download failure.
#
# The trap guarantees the final log upload + shutdown still happen on ANY
# exit (success, a set -e failure, or an uncaught error) -- without it, a
# failure would exit before reaching the log-upload/shutdown lines at the
# bottom, leaving no uploaded log AND a VM that never stops billing.
on_exit() {
  local exit_code=$?
  log "=== Script exiting with code $exit_code ==="
  kill "$LOG_UPLOADER_PID" 2>/dev/null || true
  gcloud storage cp /var/log/pisces-run.log "gs://${GCS_BUCKET}/logs/${INSTANCE_NAME}.log" || \
    log "WARNING: final log upload failed -- log only survives via serial port output while the VM is still running"
  log "Shutting down to stop billing (disk persists until manually deleted)."
  shutdown -h now
}
trap on_exit EXIT

# Periodic log upload in the background, so progress is visible on GCS even
# mid-run, not just after completion -- same rationale as Modal's periodic
# checkpoint commits.
( while true; do
    sleep 120
    gcloud storage cp /var/log/pisces-run.log "gs://${GCS_BUCKET}/logs/${INSTANCE_NAME}.log" 2>/dev/null || true
  done ) &
LOG_UPLOADER_PID=$!

log "Installing Python 3.12 (requirements.txt requires exactly 3.12, not the base image's default)..."
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

log "Pulling code from GCS..."
mkdir -p /opt/pisces
cd /opt/pisces
gcloud storage cp "gs://${GCS_BUCKET}/pisces-code.tar.gz" .
tar xzf pisces-code.tar.gz

log "Setting up venv and installing dependencies..."
python3.12 -m venv /opt/pisces/venv
source /opt/pisces/venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

export HF_TOKEN
mkdir -p ~/.cache/huggingface
echo -n "$HF_TOKEN" > ~/.cache/huggingface/token

log "Starting discover.py..."
cd /opt/pisces/track_a_feature_discovery
python discover.py --device cuda --concept "$CONCEPT" --layers $LAYERS \
  --reduced --debug-log-noop-edits --push-to-hub

log "=== discover.py finished successfully ==="
# on_exit (registered via trap above) handles the final log upload and
# shutdown for every exit path, success or failure -- nothing more needed here.
