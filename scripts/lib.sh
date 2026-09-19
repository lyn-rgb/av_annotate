#!/usr/bin/env bash
#
# One thing the other scripts share: making `curl` use the proxy this machine
# is already configured for.
#
# Why it is needed.  On macOS the system proxy lives in System Preferences and
# is read by anything using CFNetwork -- Safari, and Python's `requests` via
# `_scproxy`.  **Homebrew's curl does not read it.**  So on a Mac behind
# Clash/Surge/whoever, `pip download` and `gdown` work while an identical
# `curl` invocation sits there timing out, which looks exactly like the network
# being down when it is not.
#
# Setting HTTPS_PROXY from the system settings costs six lines and removes a
# failure that is very hard to read from the outside.
#
# Sourced, not executed.  Deliberately silent when there is nothing to do.

configure_proxy() {
    # Anything already exported wins: an operator who set these meant it.
    if [[ -n "${HTTPS_PROXY:-}${https_proxy:-}${ALL_PROXY:-}${all_proxy:-}" ]]; then
        return 0
    fi
    [[ "$(uname -s)" == "Darwin" ]] || return 0
    command -v scutil >/dev/null 2>&1 || return 0

    local enabled host port
    enabled="$(scutil --proxy 2>/dev/null | awk '/HTTPSEnable/{print $3}')"
    host="$(scutil --proxy 2>/dev/null | awk '/HTTPSProxy/{print $3}')"
    port="$(scutil --proxy 2>/dev/null | awk '/HTTPSPort/{print $3}')"
    if [[ "$enabled" == "1" && -n "$host" && -n "$port" ]]; then
        export HTTPS_PROXY="http://$host:$port"
        export HTTP_PROXY="$HTTPS_PROXY"
        # Local addresses must not go through it, or a loopback mirror -- which
        # is how the offline bundle is tested -- stops working.
        export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
        export no_proxy="$NO_PROXY"
        return 0
    fi
    return 0
}

configure_proxy
