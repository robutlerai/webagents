"""Fixture script: prints what it was asked to fill. Run by both SDKs' tests."""

import sys

print("filled " + " ".join(sys.argv[1:]))
