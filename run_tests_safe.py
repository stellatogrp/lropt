#!/usr/bin/env python
"""Run pytest with memory limits to prevent OOM crashes."""
import resource
import sys

# Limit resident set size to 24 GB (half of 48 GB system)
MAX_RAM_BYTES = 24 * 1024 * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_RSS, (MAX_RAM_BYTES, MAX_RAM_BYTES))
except (ValueError, OSError):
    # Fallback: some platforms may not support RLIMIT_RSS either
    pass

import pytest
sys.exit(pytest.main(sys.argv[1:]))
