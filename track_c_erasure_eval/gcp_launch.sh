#!/bin/bash
# Launches a Track C erasure-eval run on a GCP Compute Engine VM with an L4
# GPU. Mirrors track_a_feature_discovery/gcp_launch.sh -- see that file's
# comments for the full rationale (boot-time systemd service, multi-zone
# fallback for volatile L4 stock, per-project 1-GPU quota).
#
# Usage:
#   ./gcp_launch.sh "Golf,Uranium,Poison"
#   ./gcp_launch.sh "Homo Sapiens"
#   ./gcp_launch.sh "Golf" 0.8 10 20   # optional k, value, max-features
#                                      # overrides (see run_erasure_eval.py
#                                      # --k/--value/--max-features --
#                                      # default 0.4/36/unlimited is the
#                                      # notebook's hardcoded setting with all
#                                      # Track A-selected features, observed
#                                      # too aggressive: real model output
#                                      # collapses into repetition regardless
#                                      # of k/value once 46-435 features are
#                                      # edited simultaneously, vs the
#                                      # notebook's own validated 5-feature
#                                      # Harry Potter example)
#
# Concepts are comma-separated (not space-separated) so multi-word concept
# names survive intact through instance metadata and the startup script's
# parsing -- see gcp_startup.sh's IFS=',' read.
#
# One-time setup already done:
#   - GCS bucket gs://pisces-track-a-code-2 (same one Track A uses)
#   - pisces-code-track-c.tar.gz uploaded there -- includes artifacts/features/
#     (Track A's completed per-concept output, which run_erasure_eval.py needs
#     to read). Re-run the tar/upload step below if either the code or the
#     set of completed Track A concepts has changed since.
#
# Gemini calls go through Vertex AI, authenticated via the VM's own default
# service account (granted roles/aiplatform.user) -- no API key needed.
set -euo pipefail

CONCEPTS="${1:?Usage: ./gcp_launch.sh \"Concept One,Concept Two,...\" [k] [value] [max-features]}"
K="${2:-0.4}"
VALUE="${3:-36}"
MAX_FEATURES="${4:-0}"  # 0 means unset/unlimited -- see gcp_startup.sh

PROJECT="steering-505317"
ZONES=(us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east1-b us-east1-c us-east1-d us-east4-a)
BUCKET="pisces-track-a-code-2"
INSTANCE_NAME="pisces-trackc-$(date +%s)"
HF_TOKEN_VALUE="$(cat ~/.cache/huggingface/token)"

echo "Launching $INSTANCE_NAME for concepts='$CONCEPTS'..."

# gcloud's --metadata flag itself uses commas to separate different key=value
# pairs, so a multi-concept CONCEPTS value ("Golf,Uranium,...") gets
# misparsed as more keys ("Bad syntax for dict arg: [Uranium]") once more
# than one concept is passed -- never triggered before tonight since every
# prior launch was single-concept. --metadata-from-file reads the raw file
# content as one opaque value, sidestepping the comma-splitting entirely.
CONCEPTS_FILE="$(mktemp)"
printf '%s' "$CONCEPTS" > "$CONCEPTS_FILE"
trap 'rm -f "$CONCEPTS_FILE"' EXIT

CREATED=0
for ZONE in "${ZONES[@]}"; do
  echo "Trying zone $ZONE..."
  if gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type=g2-standard-8 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=ml-images \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --metadata-from-file=startup-script=gcp_startup.sh,concepts="$CONCEPTS_FILE" \
    --metadata=gcs-bucket="$BUCKET",hf-token="$HF_TOKEN_VALUE",k="$K",value="$VALUE",max-features="$MAX_FEATURES" \
    --scopes=https://www.googleapis.com/auth/cloud-platform 2>&1 | tee /tmp/gcp_create_attempt_c.log; then
    CREATED=1
    break
  fi
  if ! grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/gcp_create_attempt_c.log; then
    echo "Failed for a reason other than stock exhaustion -- stopping instead of trying more zones."
    exit 1
  fi
done

if [ "$CREATED" -ne 1 ]; then
  echo "Could not find L4 capacity in any known zone. Try again shortly -- stock fluctuates."
  exit 1
fi

echo ""
echo "Instance created: $INSTANCE_NAME in zone $ZONE"
echo "Watch progress (no SSH needed):"
echo "  gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE --project=$PROJECT | tail -50"
echo "Or tail the log directly once it starts uploading:"
echo "  gcloud storage cat gs://$BUCKET/logs/$INSTANCE_NAME.log --project=$PROJECT | tail -50"
echo "The VM shuts itself down (but is not deleted) once run_erasure_eval.py finishes."
