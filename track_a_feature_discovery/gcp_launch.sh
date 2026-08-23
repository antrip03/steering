#!/bin/bash
# Launches one Track A production run on a GCP Compute Engine VM with an L4
# GPU. Supplements (doesn't replace) the Modal pipeline -- see modal_app.py --
# for when Modal's budget is the constraint rather than compute availability.
#
# The VM runs gcp_startup.sh as a boot-time systemd service, entirely
# independent of this machine's network/power state once created: closing
# this laptop or losing internet does not stop the remote run, the same
# guarantee Modal's --detach gives on that platform.
#
# Usage:
#   ./gcp_launch.sh "Golf" "3 4 5 6 7 8 9 10 11 12"
#
# One-time setup already done this session:
#   - gcloud auth: already authenticated as anshultripathi002@gmail.com
#   - Compute Engine API: already enabled
#   - GCS bucket gs://pisces-track-a-code: already created and holds a
#     packaged copy of this repo (re-run the tar/upload step below if the
#     code has changed since)
#
# Quota note: this project's NVIDIA_L4_GPUS quota is 1 (region us-central1,
# confirmed via `gcloud compute regions describe us-central1`) -- only ONE
# of these VMs can run at a time without requesting a quota increase first.
set -euo pipefail

CONCEPT="${1:?Usage: ./gcp_launch.sh CONCEPT \"LAYER LAYER ...\"}"
LAYERS="${2:?Usage: ./gcp_launch.sh CONCEPT \"LAYER LAYER ...\"}"

PROJECT="project-e6820050-45d3-4631-800"
# L4 stock is genuinely volatile right now (observed us-central1-a and
# us-central1-b each reject with ZONE_RESOURCE_POOL_EXHAUSTED while pointing
# at the other as having capacity, seconds apart) -- try every zone known to
# offer L4 (confirmed via `gcloud compute accelerator-types list
# --filter="name=nvidia-l4"`) instead of hardcoding one that may be out of
# stock by the time this runs.
ZONES=(us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east1-b us-east1-c us-east1-d us-east4-a)
BUCKET="pisces-track-a-code"
INSTANCE_NAME="pisces-$(echo "$CONCEPT" | tr '[:upper:] ' '[:lower:]-')-$(date +%s)"
HF_TOKEN_VALUE="$(cat ~/.cache/huggingface/token)"

echo "Launching $INSTANCE_NAME for concept='$CONCEPT' layers='$LAYERS'..."

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
    --metadata-from-file=startup-script=gcp_startup.sh \
    --metadata=concept="$CONCEPT",layers="$LAYERS",gcs-bucket="$BUCKET",hf-token="$HF_TOKEN_VALUE" \
    --scopes=storage-full 2>&1 | tee /tmp/gcp_create_attempt.log; then
    CREATED=1
    break
  fi
  if ! grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/gcp_create_attempt.log; then
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
echo "  gcloud storage cat gs://$BUCKET/logs/$INSTANCE_NAME.log | tail -50"
echo "The VM shuts itself down (but is not deleted) once discover.py finishes and the result is pushed to the HF hub."
