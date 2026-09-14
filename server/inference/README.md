# Running the inference server

`serve.sh` runs it in the foreground and writes no log of its own. Whatever
starts it owns the output and owns restarting it. This directory offers a
supervisor for when that "whatever" should not be a terminal.

    serve.sh                    the server, configured, foreground
    spear-inference.service     a systemd template for supervising it
    install-llamacpp.sh         builds the llama.cpp binary serve.sh runs

## Configuration comes first

The unit contains **no serving decision** — no model, no port, no context
size, no card. All of those live in configuration that `serve.sh` reads:

    $SPEAR_SERVER_ROOT/config/server.conf      (default: ~/spear-runtime)
    $SPEAR_SERVER_ROOT/config/gpu.conf

See `server/config/server.conf.example` and `gpu.conf.example`. Get the server
running in the foreground first; supervise it only once it serves.

## Installing as a user unit

```sh
mkdir -p ~/.config/systemd/user
cp spear-inference.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now spear-inference
```

Copy it; do not symlink into the checkout. A unit that lives in a git working
tree changes under the supervisor on every pull.

## Persistence — read this before trusting `enable`

A user unit is supervised only while your systemd **user manager** is running.
Without lingering that manager stops with your last session and does not start
at boot, so `enable` will not survive a reboot however enabled it looks:

```sh
loginctl show-user "$USER" -p Linger
```

`Linger=no` means you have two honest choices:

| | what it does | who decides |
|---|---|---|
| `loginctl enable-linger "$USER"` | the user manager runs from boot | changes how the machine treats the account — on a shared host, its administrator |
| install as a **system** unit | supervised by pid 1 | needs root |

For a system unit, add `User=` and `Group=`, replace every `%h` with the
account's home (systemd does not expand `%h` the same way there), and install
under `/etc/systemd/system/`.

Neither is a detail of deploying a model server. Pick deliberately.

## Readiness is not startedness

`systemctl start` returns once the process has been executed, not once the
weights are loaded — for a large quantised model that is minutes. `active`
means started. To wait for serving:

```sh
until curl -sf http://127.0.0.1:<port>/health >/dev/null; do sleep 2; done
```

`llama-server` speaks no readiness protocol, so `Type=notify` is unavailable
and a unit claiming it would be lying.

## Operating it

```sh
systemctl --user status  spear-inference
systemctl --user restart spear-inference
systemctl --user stop    spear-inference
journalctl --user -u spear-inference -f
journalctl --user -u spear-inference --since "10 min ago"
```

Logging goes to the journal (`SyslogIdentifier=spear-inference`), not to a
file the service has to rotate.

## Restart policy

`Restart=on-failure`, not `always`: a clean stop is an operator's decision and
the supervisor must not undo it. `RestartSec=15`, with `StartLimitBurst=5` in
600 s so a server that cannot allocate its card stops retrying instead of
hammering the GPU. `TimeoutStartSec=900`, because a start timeout shorter than
the model load turns a working server into a restart loop.
