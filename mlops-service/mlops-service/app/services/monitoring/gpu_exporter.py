#!/usr/bin/env python3
"""GPU metrics exporter using nvidia-smi for Prometheus."""

import re
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

GPU_METRICS = """# HELP nvidia_gpu_utilization GPU utilization percentage
# TYPE nvidia_gpu_utilization gauge
# HELP nvidia_gpu_memory_used_bytes GPU memory used in bytes
# TYPE nvidia_gpu_memory_used_bytes gauge
# HELP nvidia_gpu_memory_total_bytes GPU total memory in bytes
# TYPE nvidia_gpu_memory_total_bytes gauge
# HELP nvidia_gpu_temperature_celsius GPU temperature in Celsius
# TYPE nvidia_gpu_temperature_celsius gauge
# HELP nvidia_gpu_power_watts GPU power usage in watts
# TYPE nvidia_gpu_power_watts gauge
# HELP nvidia_gpu_power_limit_watts GPU power limit in watts
# TYPE nvidia_gpu_power_limit_watts gauge
# HELP nvidia_gpu_fan_speed_percent GPU fan speed percentage
# TYPE nvidia_gpu_fan_speed_percent gauge
# HELP nvidia_gpu_clock_sm_mhz GPU SM clock speed in MHz
# TYPE nvidia_gpu_clock_sm_mhz gauge
# HELP nvidia_gpu_clock_memory_mhz GPU memory clock speed in MHz
# TYPE nvidia_gpu_clock_memory_mhz gauge
"""


def parse_mem(mem_str: str) -> int:
    """Convert memory string like '197MiB' or '197' to bytes."""
    mem_str = mem_str.strip()
    if mem_str in ('[N/A]', '', 'N/A'):
        return 0
    match = re.match(r'([\d.]+)\s*([KMGT]?i?B)?', mem_str)
    if not match:
        return 0
    value = float(match.group(1))
    unit = match.group(2) or 'MiB'
    multipliers = {
        'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4,
        'KB': 1000, 'MB': 1000**2, 'GB': 1000**3, 'TB': 1000**4,
    }
    return int(value * multipliers.get(unit, 1))


def parse_float(val_str: str) -> float:
    """Parse float value, handling [N/A]."""
    val_str = val_str.strip()
    if val_str in ('[N/A]', '', 'N/A', '-'):
        return 0.0
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def get_gpu_metrics() -> str:
    """Query nvidia-smi and return Prometheus-formatted metrics."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm,clocks.mem',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        return "# GPU metrics unavailable\n"

    lines = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 12:
            continue

        idx, name, util_gpu, util_mem, mem_used, mem_total, temp, power, power_limit, fan, clock_sm, clock_mem = parts
        name_escaped = name.replace(' ', '_').replace('-', '_')

        lines.append(f'nvidia_gpu_utilization{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(util_gpu)}')
        lines.append(f'nvidia_gpu_memory_used_bytes{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_mem(mem_used)}')
        lines.append(f'nvidia_gpu_memory_total_bytes{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_mem(mem_total)}')
        lines.append(f'nvidia_gpu_temperature_celsius{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(temp)}')
        lines.append(f'nvidia_gpu_power_watts{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(power)}')
        lines.append(f'nvidia_gpu_power_limit_watts{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(power_limit)}')
        lines.append(f'nvidia_gpu_fan_speed_percent{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(fan)}')
        lines.append(f'nvidia_gpu_clock_sm_mhz{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(clock_sm)}')
        lines.append(f'nvidia_gpu_clock_memory_mhz{{gpu="{idx}",gpu_name="{name_escaped}"}} {parse_float(clock_mem)}')

    return GPU_METRICS + '\n'.join(lines) + '\n'


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = get_gpu_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    port = int(os.environ.get('GPU_EXPORTER_PORT', 9103))
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    print(f"GPU exporter listening on http://0.0.0.0:{port}/metrics")
    server.serve_forever()


if __name__ == '__main__':
    import os
    main()
