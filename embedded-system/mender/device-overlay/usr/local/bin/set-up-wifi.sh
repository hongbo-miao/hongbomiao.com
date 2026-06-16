#!/bin/sh
set -e

raspi-config nonint do_wifi_country __WIFI_COUNTRY__ 2>/dev/null || true
rfkill unblock wifi 2>/dev/null || true

install -d -m 0755 /etc/NetworkManager/system-connections
cat > /etc/NetworkManager/system-connections/__WIFI_SSID__.nmconnection <<'NMCONN'
[connection]
id=__WIFI_SSID__
type=wifi
autoconnect=true
autoconnect-priority=10

[wifi]
mode=infrastructure
ssid=__WIFI_SSID__

[wifi-security]
key-mgmt=wpa-psk
psk=__WIFI_PSK__

[ipv4]
method=auto

[ipv6]
method=auto
NMCONN
chown root:root /etc/NetworkManager/system-connections/__WIFI_SSID__.nmconnection
chmod 0600 /etc/NetworkManager/system-connections/__WIFI_SSID__.nmconnection

nmcli radio wifi on 2>/dev/null || true
nmcli connection reload 2>/dev/null || true
nmcli connection up __WIFI_SSID__ 2>/dev/null || true
