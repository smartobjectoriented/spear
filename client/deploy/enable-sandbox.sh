#!/bin/bash
# Unblock the bubblewrap sandbox on an Ubuntu 24.04+ machine.
# Needs root. Run it ON the target machine:
#     sudo bash <checkout>/deploy/enable-sandbox.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo"; exit 1; }

echo "== before =="
echo "  apparmor_restrict_unprivileged_userns = $(sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>/dev/null || echo '(absent)')"
printf "  bwrap: "; sudo -u "${SUDO_USER:-nobody}" bwrap --dev-bind / / true 2>&1 && echo "already working"

echo "== installing the profile =="
install -m 0644 "$HERE/apparmor-bwrap" /etc/apparmor.d/bwrap
apparmor_parser -r /etc/apparmor.d/bwrap
echo "  /etc/apparmor.d/bwrap installed and loaded"

echo "== after =="
printf "  plain bwrap  : "; sudo -u "${SUDO_USER:-nobody}" bwrap --dev-bind / / true 2>&1 && echo OK
printf "  bwrap netns  : "; sudo -u "${SUDO_USER:-nobody}" bwrap --unshare-net --dev-bind / / true 2>&1 && echo OK
echo
echo "Then verify, as a normal user:"
echo "  cd <checkout> && ./bin/python -m unittest discover -s tests"
