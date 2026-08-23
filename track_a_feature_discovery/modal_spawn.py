"""
Fire-and-forget launcher for the deployed pisces-track-a Modal app --
genuinely immune to local network drops, unlike `modal run --detach`.

Why this exists: `modal run --detach` only protects against the local CLI
process being killed or cleanly disconnecting -- it does NOT protect against
a DNS resolution failure inside the CLI's own heartbeat loop
([Errno 11001] getaddrinfo failed), which happened twice this session and
stopped two detached runs outright (one 66% through a ~2h effect-measurement
stage). `.spawn()` on a deployed app only needs the local connection for the
brief initial submit call -- once Modal accepts the job, it's running
entirely on Modal's infrastructure with no further dependency on this
machine's network at all.

One-time setup (re-run whenever modal_app.py changes):
    modal deploy track_a_feature_discovery/modal_app.py

Usage:
    python modal_spawn.py --concept Golf --layers 3,4,5,6,7,8,9,10,11,12 --reduced
    python modal_spawn.py --concept "Golf,Uranium" --candidates-only

Prints a call_id -- save it to check on the job later:
    python modal_spawn.py --status <call_id>
"""
from __future__ import annotations

import argparse
import sys

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", help="Comma-separated concept name(s).")
    parser.add_argument("--layers", default="", help="Comma-separated layer numbers.")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--minmatch", type=int, default=None)
    parser.add_argument("--enable-cascade", action="store_true")
    parser.add_argument("--cvs-path", default=None)
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--features", default="", help="Comma-separated layer:id:neg specs.")
    parser.add_argument("--corpus-batches", type=int, default=None)
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument("--no-debug-log-noop-edits", action="store_true")
    parser.add_argument("--status", metavar="CALL_ID", help="Check on a previously spawned call instead of launching a new one.")
    args = parser.parse_args()

    if args.status:
        call = modal.FunctionCall.from_id(args.status)
        try:
            result = call.get(timeout=0)
            print(f"FINISHED: discover.py exit code {result}")
        except TimeoutError:
            print("Still running.")
        return

    if not args.concept:
        parser.error("--concept is required unless using --status")

    f = modal.Function.from_name("pisces-track-a", "run_discover")
    call = f.spawn(
        concepts=[c.strip() for c in args.concept.split(",") if c.strip()],
        layers=[int(x) for x in args.layers.split(",") if x.strip()] or None,
        reduced=args.reduced,
        minmatch=args.minmatch,
        enable_cascade=args.enable_cascade,
        cvs_path=args.cvs_path,
        candidates_only=args.candidates_only,
        features=[f.strip() for f in args.features.split(",") if f.strip()] or None,
        corpus_batches=args.corpus_batches,
        debug_log_noop_edits=not args.no_debug_log_noop_edits,
        push_to_hub=not args.no_push_to_hub,
    )
    print(f"Spawned. call_id={call.object_id}")
    print(f"Check status later with: python modal_spawn.py --status {call.object_id}")
    print(f"Or view logs at: https://modal.com/apps/anshul-t/main/deployed/pisces-track-a")


if __name__ == "__main__":
    main()
