#!/bin/bash
#
# RetinaXAI Monitoring Setup Script
# Configures Prometheus and Grafana for local monitoring
#
# Usage: sudo ./setup-monitoring.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETINAXAI_BASE="/home/louay/RetinaXAI"
MONITORING_DIR="${RETINAXAI_BASE}/infra/infra/monitoring"

# System paths
PROMETHEUS_CONFIG_DIR="/etc/prometheus"
GRAFANA_PROVISIONING_DIR="/etc/grafana/provisioning"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RetinaXAI Monitoring Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Check if monitoring directory exists
if [ ! -d "$MONITORING_DIR" ]; then
    echo -e "${RED}Error: Monitoring directory not found: $MONITORING_DIR${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Backing up existing configurations...${NC}"

# Backup Prometheus config
if [ -f "${PROMETHEUS_CONFIG_DIR}/prometheus.yml" ]; then
    sudo cp "${PROMETHEUS_CONFIG_DIR}/prometheus.yml" "${PROMETHEUS_CONFIG_DIR}/prometheus.yml.bak.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓ Backed up Prometheus config${NC}"
fi

# Backup Grafana datasources
if [ -d "${GRAFANA_PROVISIONING_DIR}/datasources" ]; then
    sudo cp -r "${GRAFANA_PROVISIONING_DIR}/datasources" "${GRAFANA_PROVISIONING_DIR}/datasources.bak.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓ Backed up Grafana datasources${NC}"
fi

# Backup Grafana dashboards
if [ -d "${GRAFANA_PROVISIONING_DIR}/dashboards" ]; then
    sudo cp -r "${GRAFANA_PROVISIONING_DIR}/dashboards" "${GRAFANA_PROVISIONING_DIR}/dashboards.bak.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓ Backed up Grafana dashboards${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Copying Prometheus configuration...${NC}"

# Copy Prometheus config
sudo cp "${MONITORING_DIR}/prometheus.yml" "${PROMETHEUS_CONFIG_DIR}/prometheus.yml"
echo -e "${GREEN}✓ Copied prometheus.yml${NC}"

# Copy alert rules
if [ -f "${MONITORING_DIR}/alert-rules.yml" ]; then
    sudo cp "${MONITORING_DIR}/alert-rules.yml" "${PROMETHEUS_CONFIG_DIR}/alert-rules.yml"
    echo -e "${GREEN}✓ Copied alert-rules.yml${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Copying Grafana provisioning files...${NC}"

# Copy Grafana datasource
sudo cp "${MONITORING_DIR}/grafana/datasources/prometheus.yml" "${GRAFANA_PROVISIONING_DIR}/datasources/prometheus.yml"
echo -e "${GREEN}✓ Copied Prometheus datasource${NC}"

# Copy dashboards
sudo cp "${MONITORING_DIR}/grafana/dashboards/"*.json "${GRAFANA_PROVISIONING_DIR}/dashboards/"
echo -e "${GREEN}✓ Copied dashboard files${NC}"

echo ""
echo -e "${YELLOW}Step 4: Validating configurations...${NC}"

# Validate Prometheus config
if command -v promtool &> /dev/null; then
    if promtool check config "${PROMETHEUS_CONFIG_DIR}/prometheus.yml" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Prometheus config is valid${NC}"
    else
        echo -e "${RED}✗ Prometheus config validation failed${NC}"
        promtool check config "${PROMETHEUS_CONFIG_DIR}/prometheus.yml"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ promtool not found, skipping validation${NC}"
fi

echo ""
echo -e "${YELLOW}Step 5: Restarting services...${NC}"

# Restart Prometheus
if systemctl is-active --quiet prometheus; then
    sudo systemctl restart prometheus
    echo -e "${GREEN}✓ Restarted Prometheus${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus service not found or not running${NC}"
fi

# Restart Grafana
if systemctl is-active --quiet grafana-server; then
    sudo systemctl restart grafana-server
    echo -e "${GREEN}✓ Restarted Grafana${NC}"
else
    echo -e "${YELLOW}⚠ Grafana service not found or not running${NC}"
fi

echo ""
echo -e "${YELLOW}Step 6: Installing GPU exporter service...${NC}"

# Install GPU exporter systemd service
if [ -f "${MONITORING_DIR}/gpu-exporter.service" ]; then
    sudo cp "${MONITORING_DIR}/gpu-exporter.service" /etc/systemd/system/retinaxai-gpu-exporter.service
    sudo systemctl daemon-reload
    sudo systemctl enable retinaxai-gpu-exporter
    sudo systemctl start retinaxai-gpu-exporter
    echo -e "${GREEN}✓ GPU exporter service installed and started${NC}"
else
    echo -e "${YELLOW}⚠ GPU exporter service file not found${NC}"
fi

echo ""
echo -e "${YELLOW}Step 7: Verifying services...${NC}"

# Wait for services to start
sleep 3

# Check Prometheus
if curl -s http://localhost:9090/api/v1/status > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus is running (port 9090)${NC}"
else
    echo -e "${RED}✗ Prometheus is not responding${NC}"
fi

# Check Grafana
if curl -s http://localhost:4000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Grafana is running (port 4000)${NC}"
else
    echo -e "${RED}✗ Grafana is not responding${NC}"
fi

# Check Node Exporter
if curl -s http://localhost:9100/metrics > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Node Exporter is running (port 9100)${NC}"
else
    echo -e "${YELLOW}⚠ Node Exporter is not responding (install if needed)${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Access URLs:${NC}"
echo -e "  Prometheus:     ${BLUE}http://localhost:9090${NC}"
echo -e "  Grafana:        ${BLUE}http://localhost:4000${NC} (admin/prtgrm1998)"
echo -e "  MLOps Dashboard: ${BLUE}http://localhost:4000/d/retinaxai-mlops-dashboard${NC}"
echo -e "  System Dashboard: ${BLUE}http://localhost:4000/d/retinaxai-system-metrics${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Login to Grafana (admin/prtgrm1998)"
echo "  2. Navigate to Dashboards to view RetinaXAI metrics"
echo "  3. Check Prometheus Targets at http://localhost:9090/targets"
echo "  4. Start MLOps service to see ML metrics"
echo "  5. Check GPU exporter at http://localhost:9103/metrics"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo "  - View logs: sudo journalctl -u prometheus -f"
echo "  - View logs: sudo journalctl -u grafana-server -f"
echo "  - Check targets: curl http://localhost:9090/api/v1/targets | jq"
echo ""
