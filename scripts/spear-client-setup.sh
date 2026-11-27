#!/bin/bash
# Prepares a workstation to run the REDS SPEAR project image: trusts the
# registry, logs in, pulls the image and installs the spear launcher.
#
# Needs docker (your user in the docker group), sudo once for the
# certificate, the HEIG-VD VPN and the registry's pull password. Run it
# again to update the image:  spear-client-setup.sh [image]

set -e

REG=reds-ml-2.reds-heig-vd.in:5000
IMAGE=${1:-$REG/spear:0.3.0-project}

sudo mkdir -p /etc/docker/certs.d/$REG
sudo tee /etc/docker/certs.d/$REG/ca.crt >/dev/null <<'CRT'
-----BEGIN CERTIFICATE-----
MIIFYjCCA0qgAwIBAgIUHRcg/cuV348HP8saz7FbXZndy8UwDQYJKoZIhvcNAQEL
BQAwJDEiMCAGA1UEAwwZcmVkcy1tbC0yLnJlZHMtaGVpZy12ZC5pbjAeFw0yNjEx
MjYxNTQyMTdaFw0zNjExMjMxNTQyMTdaMCQxIjAgBgNVBAMMGXJlZHMtbWwtMi5y
ZWRzLWhlaWctdmQuaW4wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDg
VKJiU3/nUxD3WJY2qjP3bD95gIYG0yUkHrg8CjoUQhZOPaSohOgbkCrxLL0Qkjdr
xoll/rodGgv++EgxcKV5blTqM3f6LIQqYWgawTyXUBxelebWQ9X6acChETjIhVQe
2uYEM9fFuyIjo5KhSdYWUlpyTShZL6xgPOCdGJtG6EUmLXww1Sl7rWJec+dzXpz8
HU9vUEELcDXL7OXogU0xOouiN0tD/ZIMLPaqQSieaBjjjmefc/RNBintTRUBoDI6
YD0xR3syLnm+8iEnnUc8NP7rT/gUl5DXWht+hTVyntR3uE8dsEr+rAm5KMkhBhhl
bWLsPRBn9ypgRMAwe1AJ+egk9TSsLu1Fw8Gbge6Dx2BTGMGLFeo4s6hQYkuMfu0K
aQgCO56Fff5MkrZdzwvX4D9ZqIc5e9PVbUTeosbJ2p63I0oNZP1k3e0cN1YlhOjc
hZRyDuvfqlHbyM3RNwrx1/vmx33T4SyotiGd0Tql1+DNyElzEIH9amGfKqFTM5pM
G+TaQDsyLBCaqOTHcDQ9wGDD9O6uBhwqRh5brwY+HeGNbmIEhKE69kBX5nhFbhSV
pxDWw8cFwANPM6e5rAOZAKQ9731CVDx3dRP4Vv34m74DMdKDIaQj9u8Ubf5pPqyD
KsX0Ng3nTmJLHvcdrPLJiO+oNv0NEMyKd3eqtYuBHwIDAQABo4GLMIGIMB0GA1Ud
DgQWBBQsWcLBLmsm6dfZzNjcRO8UIv1AazAfBgNVHSMEGDAWgBQsWcLBLmsm6dfZ
zNjcRO8UIv1AazA1BgNVHREELjAsghlyZWRzLW1sLTIucmVkcy1oZWlnLXZkLmlu
gglyZWRzLW1sLTKHBAq+pNowDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsF
AAOCAgEAU6l9Tx45TS1Rt0xxvILvYDtR1fGFMqeTQvTe5G/sxpaHctXPWwwmtlCL
p7f1nr9uB/dm1rfYLm/X32atD0Q0fbCuq36honGLWwVqngUmjaQ2vB15mgVKSjyz
xNjlo+uH2qHBzGajnQM7eB4J9fXNdpXJ2IWn309cBBrajMNPgJ1Wfrsgt0pOiGWY
u1oltWN8yCXD6qeyaz6QBzyhUUh6bMPD5a8YttWImMRa8VtGOcDLDzoqr8GB8W4A
z7xCq1cuevEJL0HZPY5f8X3AdGvUrOSciTythisuQJ8zFhQA4EliNMKVG+pDMuoJ
LLB1HNs7cxLuEqMjxqzB1amGG/r26CsvlgRL4CYCMi1cqm6Lqr6SI1qgvC25sf+p
EmqgRM6wAGcLZbdYCg3lbwdYL6NKOVMBqwVZT67uq5fib0CkLzbHg/VEzqM3/LSF
CdwK5q2ZCmIHjmB7nT7HSkr5/ZSFoXkmBe3yDmB6LbR+FIvrXmAd3qhGm/rrpzD2
+diZoht+8AFHHLgNZ/bevTeMnOK7Y4EKSEDIzqgIKFBltyBAfHfP3y6M8Nuew0/x
421W3CLHwEGDDyiJJRQphjtcSwrvA19qFN0vxPKLwbszLFWQLzBPT/LDuFvTp3lB
aRNXrE75DvbQ1A8fzkWW5+ckvWso1tvECsRcple7qUWte8dbqsI=
-----END CERTIFICATE-----
CRT

# The registry's pull password is not in this file: ask Daniel for it, then
# give it in SPEAR_REGISTRY_PASSWORD or type it when asked.

if [ -n "${SPEAR_REGISTRY_PASSWORD:-}" ]; then
    printf '%s' "$SPEAR_REGISTRY_PASSWORD" | docker login "$REG" -u spear --password-stdin
else
    docker login "$REG" -u spear
fi
docker pull "$IMAGE"

# The launcher travels in the image; --install copies it to ~/.local/bin and
# writes ~/.config/spear/local.conf naming the image, unless one exists.

tmp=$(mktemp)
docker run --rm --entrypoint cat "$IMAGE" /opt/spear/spear > "$tmp"
bash "$tmp" --install --image "$IMAGE"
rm -f "$tmp"

# The configuration of opencn-ng/spear-private/workstation.conf: an existing
# file that says something else is kept as local.conf.bak.

CONF="${XDG_CONFIG_HOME:-$HOME/.config}/spear/local.conf"
new=$(mktemp)
cat > "$new" <<EOF
# Read by spear before its options; shell, one assignment per line.
SPEAR_IMAGE=$IMAGE

# The model spear-chat talks to: the REDS inference host, through an ssh
# tunnel opened with this key alone (the host's MaxAuthTries is low).

REDS_HOST=redsroot@reds-ml-2.reds-heig-vd.in
SPEAR_REDS_KEY="\$HOME/.ssh/id_ed25519"
EOF
if ! cmp -s "$new" "$CONF"; then
    [ -e "$CONF" ] && cp "$CONF" "$CONF.bak" && echo "kept the former $CONF as $CONF.bak"
    mv "$new" "$CONF"
    echo "wrote $CONF"
else
    rm -f "$new"
fi
[ -r "$HOME/.ssh/id_ed25519.pub" ] || echo "note: no ~/.ssh/id_ed25519: run ssh-keygen -t ed25519, or set SPEAR_REDS_KEY to your key"

echo
echo "Done. Then, in a checkout or anywhere: spear     (a shell in the image)"
echo "spear-chat needs your ~/.ssh/id_ed25519.pub added on reds-ml-2 (send it to Daniel)"
