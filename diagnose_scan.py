"""Local-only sampling diagnostics. No API key or paid inference required."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Original camera video, at most 20 seconds")
    args = parser.parse_args()
    # Importing loads the existing backend dependencies, but creates no model
    # session or Anthropic client. Output contains metadata, never frame pixels.
    from video_scan import _process_video_sync
    report = []
    try:
        _process_video_sync(args.video, diagnostic_only=True, report=report)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "events": report}, indent=2))
        return 1
    print(json.dumps({"status": "done", "diagnostic_only": True,
                      "events": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
