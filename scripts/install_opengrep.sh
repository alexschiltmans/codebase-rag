#!/usr/bin/env bash
# Fetch the pinned Opengrep binary into .tools/.
#
# Opengrep publishes standalone binaries per platform rather than a Python package, so it cannot
# ride along in the dev extra like ruff and mypy do. The version is pinned here so a scan on a
# laptop and a scan in CI run the same engine: an unpinned static analyser silently changes what
# it reports, which turns a clean run into a claim nobody can reproduce.
set -euo pipefail

OPENGREP_VERSION="${OPENGREP_VERSION:-v1.26.0}"
TOOLS_DIR="${TOOLS_DIR:-.tools}"
TARGET="${TOOLS_DIR}/opengrep"

os="$(uname -s)"
arch="$(uname -m)"

case "${os}/${arch}" in
  Darwin/arm64)  asset="opengrep_osx_arm64" ;;
  Darwin/x86_64) asset="opengrep_osx_x86" ;;
  Linux/aarch64) asset="opengrep_manylinux_aarch64" ;;
  Linux/arm64)   asset="opengrep_manylinux_aarch64" ;;
  Linux/x86_64)  asset="opengrep_manylinux_x86" ;;
  *)
    echo "No Opengrep build for ${os}/${arch}." >&2
    exit 1
    ;;
esac

# An existing binary at the pinned version is left alone: the download is ~40MB and this runs
# from a Makefile target that the scan target depends on.
if [ -x "${TARGET}" ] && "${TARGET}" --version 2>/dev/null | grep -q "${OPENGREP_VERSION#v}"; then
  echo "Opengrep ${OPENGREP_VERSION} already present at ${TARGET}."
  exit 0
fi

mkdir -p "${TOOLS_DIR}"
url="https://github.com/opengrep/opengrep/releases/download/${OPENGREP_VERSION}/${asset}"
echo "Downloading ${url}"
curl -fsSL "${url}" -o "${TARGET}.tmp"
chmod +x "${TARGET}.tmp"
mv "${TARGET}.tmp" "${TARGET}"
echo "Installed Opengrep ${OPENGREP_VERSION} to ${TARGET}."
