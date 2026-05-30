#!/usr/bin/env bash
set -euo pipefail

# Enable Grafana iframe embedding + anonymous read-only access
# Run with sudo

echo "Enabling allow_embedding in Grafana config..."
sed -i '/^\[security\]/,/^\[/{s/^;allow_embedding = .*/allow_embedding = true/}' /etc/grafana/grafana.ini

echo "Enabling anonymous access..."
sed -i '/^\[auth.anonymous\]/,/^\[/{s/^;enabled = .*/enabled = true/}' /etc/grafana/grafana.ini
sed -i '/^\[auth.anonymous\]/,/^\[/{s/^;org_role = .*/org_role = Viewer/}' /etc/grafana/grafana.ini

echo "Restarting Grafana..."
systemctl restart grafana-server

echo "Done. Grafana is now embeddable via iframe."
