#!/usr/bin/env python3
"""
Generate a PEP 503 simple index for the brahe package from GitHub releases.

This script fetches the latest release assets and creates a simple index
that allows pip to install the package directly from GitHub releases without PyPI.

Usage:
    python scripts/generate_simple_index.py --repo <owner/repo> --output <dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def fetch_release_assets(repo: str, tag: str = "latest") -> list[dict]:
    """
    Fetch release assets from GitHub API.

    Args:
        repo: Repository in format "owner/repo"
        tag: Release tag (default: "latest")

    Returns:
        List of asset dictionaries with 'name' and 'browser_download_url'
    """
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    # GitHub API requires a User-Agent header
    headers = {
        "User-Agent": "brahe-simple-index-generator",
        "Accept": "application/vnd.github.v3+json",
    }

    # Add GitHub token if available (for rate limiting)
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    request = Request(api_url, headers=headers)

    try:
        with urlopen(request) as response:
            data = json.loads(response.read())
            return data.get("assets", [])
    except HTTPError as e:
        if e.code == 404:
            print(f"Release '{tag}' not found in {repo}", file=sys.stderr)
            return []
        raise


def generate_simple_index_html(package_name: str, assets: list[dict], repo: str) -> str:
    """
    Generate PEP 503 compliant simple index HTML.

    Args:
        package_name: Name of the package
        assets: List of asset dictionaries from GitHub API
        repo: Repository in format "owner/repo"

    Returns:
        HTML string for the simple index
    """
    # Filter for wheel and sdist files
    relevant_assets = [
        asset for asset in assets if asset["name"].endswith((".whl", ".tar.gz"))
    ]

    if not relevant_assets:
        print("Warning: No wheel or sdist files found in release", file=sys.stderr)

    # Generate HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"  <title>Links for {package_name}</title>",
        "</head>",
        "<body>",
        f"  <h1>Links for {package_name}</h1>",
    ]

    for asset in sorted(relevant_assets, key=lambda x: x["name"]):
        name = asset["name"]
        url = asset["browser_download_url"]
        html_parts.append(f'  <a href="{url}">{name}</a><br/>')

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PEP 503 simple index from GitHub releases"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository in format 'owner/repo'",
    )
    parser.add_argument(
        "--package-name",
        default="brahe",
        help="Package name (default: brahe)",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Release tag (default: latest)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for simple index",
    )

    args = parser.parse_args()

    # Fetch release assets
    print(f"Fetching release assets from {args.repo} (tag: {args.tag})...")
    assets = fetch_release_assets(args.repo, args.tag)

    if not assets:
        print("No assets found in release", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(assets)} assets")

    # Generate index HTML
    html = generate_simple_index_html(args.package_name, assets, args.repo)

    # Create output directory structure: output/simple/package_name/index.html
    output_path = Path(args.output) / "simple" / args.package_name
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "index.html"
    index_file.write_text(html)

    print(f"Simple index generated at: {index_file}")
    print("\nTo install with pip:")
    print(f"  pip install {args.package_name} --extra-index-url <base-url>/simple/")


if __name__ == "__main__":
    main()
