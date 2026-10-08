from __future__ import annotations

import sys

from wan_tool import main


if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
