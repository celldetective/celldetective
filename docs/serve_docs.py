#!/usr/bin/env python
"""Build the Celldetective docs and serve them on localhost.

Usage (from anywhere)::

    python docs/serve_docs.py            # build once, serve, open browser
    python docs/serve_docs.py --live     # auto-rebuild on save + live reload
    python docs/serve_docs.py --port 9000
    python docs/serve_docs.py --clean    # wipe build/ first (full rebuild)
    python docs/serve_docs.py --no-open  # don't open a browser

``--live`` needs ``sphinx-autobuild`` (``pip install sphinx-autobuild``); the
script tells you and falls back to a one-shot build + static server if it is
missing. Stop the server with Ctrl+C.
"""

import argparse
import functools
import http.server
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import webbrowser

# Paths are resolved relative to this file so the script works from any CWD.
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(DOCS_DIR, "source")
BUILD_DIR = os.path.join(DOCS_DIR, "build", "html")


def _find_free_port(preferred: int) -> int:
    """Return ``preferred`` if free, otherwise the next available port."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def build_once(clean: bool) -> None:
    """Run a one-shot HTML build with sphinx-build."""
    if clean and os.path.isdir(os.path.join(DOCS_DIR, "build")):
        shutil.rmtree(os.path.join(DOCS_DIR, "build"))
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", SOURCE_DIR, BUILD_DIR]
    print("Building docs:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            "\nSphinx build failed (see errors above). "
            "Fix the errors, or pass --clean for a full rebuild."
        )


def serve_static(port: int, open_browser: bool) -> None:
    """Serve the built HTML on 127.0.0.1:<port> until Ctrl+C."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=BUILD_DIR
    )

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    url = f"http://127.0.0.1:{port}/"
    with Server(("127.0.0.1", port), handler) as httpd:
        print(f"\nServing docs at {url}  (Ctrl+C to stop)")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def serve_live(port: int, open_browser: bool, clean: bool) -> bool:
    """Use sphinx-autobuild for rebuild-on-save + live reload.

    Returns False if sphinx-autobuild is not installed (caller falls back).
    """
    if shutil.which("sphinx-autobuild") is None:
        try:
            import sphinx_autobuild  # noqa: F401
        except ImportError:
            print(
                "--live needs sphinx-autobuild, which is not installed.\n"
                "  Install it with:  pip install sphinx-autobuild\n"
                "Falling back to a one-shot build + static server.\n"
            )
            return False

    if clean and os.path.isdir(os.path.join(DOCS_DIR, "build")):
        shutil.rmtree(os.path.join(DOCS_DIR, "build"))

    cmd = [
        sys.executable,
        "-m",
        "sphinx_autobuild",
        SOURCE_DIR,
        BUILD_DIR,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if open_browser:
        cmd.append("--open-browser")
    print("Live docs (auto-rebuild on save):", " ".join(cmd))
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve on (default: 8000)."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Auto-rebuild on save with live reload (needs sphinx-autobuild).",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove build/ before building."
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a web browser."
    )
    args = parser.parse_args()

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} busy; using {port} instead.")

    open_browser = not args.no_open

    if args.live and serve_live(port, open_browser, args.clean):
        return

    build_once(args.clean)
    serve_static(port, open_browser)


if __name__ == "__main__":
    main()
