import ZabFetch as zb
import time
import pandas as pd
import os
import json
import filecmp
from functools import reduce
from functools import reduce
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
import logging
import re
import os
from datetime import datetime
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import pdfkit
from vector_emailsender import Vector_EmailSender
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

config_path = os.path.join('config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    db_config = config['database_config']
    email_config = config.get("email_details", {})

email_sender = Vector_EmailSender(
    username=email_config.get("username"),
    password=email_config.get("password"),
    smtp_server="smtp.gmail.com",
    smtp_port=465
)

zbx= zb.ZabbixDB(**db_config)

time_to = int(time.time())
time_from = time_to - 86400*30

def setup_logger(log_file: Optional[str]) -> None:
    """
    Configure the logger to write only WARNING and ERROR messages to a file.

    Args:
        log_file (Optional[str]): Path to the log file. If None or empty, use default.

    Raises:
        ValueError: If log_file cannot be created.
    """
    logger = logging.getLogger(__name__)
    logger.handlers = []  # Clear existing handlers to avoid duplicates
    logger.setLevel(logging.WARNING)  # Only capture WARNING and ERROR

    # Set default log file in 'logs' folder if none provided
    if not log_file or log_file.isspace():
        log_file = os.path.join("logs", "resource_utilization.log")
        logger.warning(f"No valid log_file provided; using default: {log_file}")

    # Create directory for log file
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_file = os.path.join(".", log_file)

    # Create file handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to create log file {log_file}: {e}")
        raise ValueError(f"Cannot create log file {log_file}: {e}")

def get_os_data(text):
    # text = "Windows Server 2025 Datacenter 26100.1.amd64fre.ge_release.240331-1435"
    # text = "Windows 11 Pro 10.0.22000"

    windows_pattern = re.compile(r'\b(Windows(?: Server)?)\s+(\d{2,4})', re.IGNORECASE)

    # Linux distro pattern updated to capture ~22.04 correctly
    linux_distro_pattern = re.compile(
        r'\b(Ubuntu|Debian|RedHat|CentOS|Fedora|Alpine|Arch)[^\n]*?~(\d+\.\d+)',
        re.IGNORECASE
    )

    linux_pattern = re.compile(r'\bLinux\b', re.IGNORECASE)

    os_type = None
    version = None

    # Check for Windows
    win_match = windows_pattern.search(text)
    if win_match:
        os_type = win_match.group(1).strip()
        version = win_match.group(2).strip()

    else:
        # Check for Linux
        linux_match = linux_pattern.search(text)
        if linux_match:
            os_type = "Linux"
            # Look for distro and correct OS version (~22.04)
            distro_match = linux_distro_pattern.search(text)
            if distro_match:
                distro = distro_match.group(1).strip()
                distro_version = distro_match.group(2).strip()
                os_type = f"{os_type} {distro}"
                version = distro_version

    # Output
    if os_type and version:
        return f"{os_type} {version}"
    elif os_type:
        return os_type
    else:
        print("No OS information matched.")

def get_time_diff(time_span_seconds: int):
    min_bin_size: int = 5 * 60
        # Decide bin size based on time span
    if time_span_seconds < 6 * 3600:  # Less than 6 hours
        bin_seconds = max(min_bin_size, 15 * 60)  # 15 minutes
    elif time_span_seconds < 25 * 3600:  # Less than 12 hours
        bin_seconds = max(min_bin_size, 3600)  # 1 hour
    elif time_span_seconds < 7 * 24 * 3600:  # Less than 7 days
        bin_seconds = max(min_bin_size, 24 * 3600)  # 1 day
    else:
        bin_seconds = max(min_bin_size, 24 * 3600)  # 1 day
    return bin_seconds

def get_drive_name(drive: str) -> str:
    # Try to extract value inside parentheses first
    match = re.search(r'\(\{#FSNAME\}\)', drive)
    if match:
        result = '{#FSNAME}'
    else:
        # If not found, try to extract value inside square brackets
        match = re.search(r'\[([^\]]+)\]', drive)
        result = match.group(1) if match else None

    # Output: only '{#FSNAME}' if found in parentheses, otherwise value inside []
    # If neither is found, result will be None
    # Extract only the drive letter (e.g., 'C') if result is like '(C:)'
    if result and re.match(r'\([A-Za-z]:\)', result):
        drive_letter = result[1]  # character after '('
        return drive_letter
    else:
        return result

def get_interface_name(interface: str) -> str:
    # Match everything after 'Interface' up to the first colon
    match = re.search(r'Interface\s+(.*?):', interface)
    return match.group(1).strip() if match else None

def format_bytes(size_in_bytes: int) -> str:
    """
    Convert a byte size to a human-readable format (MB, GB, TB, etc.).
    
    Args:
        size_in_bytes (int): The size in bytes.
    
    Returns:
        str: Human-readable string.
    """
    if size_in_bytes < 0:
        raise ValueError("Size cannot be negative.")

    # Define units
    units = ["Bytes", "KB", "MB", "GB", "TB", "PB"]
    factor = 1024.0

    size = float(size_in_bytes)
    for unit in units:
        if size < factor:
            return f"{size:.2f} {unit}"
        size /= factor

    return f"{size:.2f} PB"

def format_interface_bytes(size_in_bytes: int) -> str:
    """
    Convert a byte size to a human-readable format for network interfaces (Kbps, Mbps, Gbps, etc.).
    
    Args:
        size_in_bytes (int): The size in bytes.
    
    Returns:
        str: Human-readable string.
    """
    if size_in_bytes < 0:
        raise ValueError("Size cannot be negative.")

    # Define units
    units = ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]
    factor = 1024.0

    size = float(size_in_bytes)
    for unit in units:
        if size < factor:
            return f"{size:.2f} {unit}"
        size /= factor

    return f"{size:.2f} Tbps"

def unixtotime(x):
    # Example: convert UNIX timestamp to readable string
    return pd.to_datetime(x, unit='s')

def seconds_to_compact_time(seconds):
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not (days or hours or minutes):  # Show seconds only if it's the only unit
        parts.append(f"{seconds}s")

    return ''.join(parts) if parts else "0s"

def duration_to_label(time_from, time_to):
    """
    Returns a compact duration string between two UNIX timestamps.
    Examples:
        45 seconds -> "45s"
        90 minutes -> "1h"
        2 days -> "2d"
        10 days -> "1w"
        45 days -> "1m"
        400 days -> "1y"
    """
    seconds = int(time_to - time_from)
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days}d"
    elif seconds < 2592000:
        weeks = seconds // 604800
        return f"{weeks}w"
    elif seconds < 31536000:
        months = seconds // 2592000
        return f"{months}m"
    else:
        years = seconds // 31536000
        return f"{years}y"

def get_host_support_data(hostname: str, metric_name: Union[str, List[str]], time_from: int, time_to: int):

    log_file = '/root/zabbix-lib/logs/resource_utilization.log'
    # Setup logger
    setup_logger(log_file)
    logger = logging.getLogger(__name__)
    
    orignal_time_from = time_from
    time_from = time_to - 87400

    cpu_items = ['CPU utilization', 'CPU idle time', 'CPU system time', 'CPU user time']
    
    groups = ['	Linux servers','Windows servers']
    
    supporting_data = {
        'hostname': hostname,
    }

    host_ip = zbx.get_host_ip(hostname)
    supporting_data['host_ip'] = host_ip['data']['ip']

    host_group = zbx.get_host_by_group(hostname=hostname)
    host_group = [g['group_name'] for g in host_group]
    host_group = list(set(groups) & set(host_group))
    supporting_data['host_group'] = host_group
    os_type = zbx.get_metric_data(
        hostname=hostname,
        metric_name='Operating system',
        time_from=time_from,
        time_to=time_to,

    )['data'][0]['value']
    supporting_data['os_type'] = get_os_data(os_type)
    interfaces = []
    for metric in metric_name:
        if metric in cpu_items:
            supporting_data['Number of CPUs'] = zbx.get_metric_data(
                hostname=hostname,
                metric_name='Number of CPUs',
                time_from=time_from,
                time_to=time_to,
                statistical_measure='last'
            )['data'][0]['value']
            if supporting_data['Number of CPUs'] is None:
                supporting_data['Number of cores'] = zbx.get_metric_data(
                    hostname=hostname,
                    metric_name='Number of cores',
                    time_from=time_from,
                    time_to=time_to,
                    statistical_measure='last'
                )['data'][0]['value']
            
            supporting_data['threshold'] = zbx.get_macro_for_host(
                hostname=hostname,
                macro_name='{$CPU.UTIL.CRIT}'
            )['data'][0]['value']

        elif metric.startswith('FS'):
            drive_name = get_drive_name(metric)
            totl_space = metric.replace('Space: Used, in %', 'Space: Total').strip()
            available_space = metric.replace('Space: Used, in %', 'Space: Available').strip()

            # Get total space data
            total_resp = zbx.get_metric_data(
                hostname=hostname,
                metric_name=totl_space,
                time_from=time_from,
                time_to=time_to,
                statistical_measure='last'
            )
            total_data = total_resp.get('data', [])
            if total_data:
                supporting_data[f'{drive_name} Total Space'] = format_bytes(total_data[0]['value'])
                
            else:
                print(f"No data for {totl_space}")

            # Get available space data
            available_resp = zbx.get_metric_data(
                hostname=hostname,
                metric_name=available_space,
                time_from=time_from,
                time_to=time_to,
                statistical_measure='last'
            )
            available_data = available_resp.get('data', [])
            if available_data:
                supporting_data[f'{drive_name} Available Space'] = format_bytes(available_data[0]['value'])
                
            else:
                logger.info(f"No data for {available_space}")
        
        elif metric.startswith('Interface'):
            interface_name = get_interface_name(metric)
            
            if interface_name not in interfaces:
                interfaces.append(interface_name)
                pattern = r"(Bits received|Bits sent|Inbound packets discarded|Inbound packets with errors|Outbound packets discarded|Outbound packets with errors)"

                interface_speed = re.sub(pattern, "Speed", metric)
                interface_speed_data = zbx.get_metric_data(
                    hostname=hostname,
                    metric_name=interface_speed,
                    time_from=time_from,
                    time_to=time_to,
                    statistical_measure='last'
                )
                if interface_speed_data['data']:
                    speed_value = interface_speed_data['data'][0]['value']
                    if speed_value is not None:
                        supporting_data[f'{interface_name} Speed'] = f"{speed_value} bps"
                
                interface_type = re.sub(pattern, "Interface type", metric)
                interface_type_data = zbx.get_metric_data(
                    hostname=hostname,
                    metric_name=interface_type,
                    time_from=time_from,
                    time_to=time_to,
                    statistical_measure='last'
                )
                if interface_type_data['data']:
                    type_value = interface_type_data['data'][0]['value']
                    if type_value is not None:
                        supporting_data[f'{interface_name} Type'] = type_value
                
                interface_operational_Status = re.sub(pattern, "Operational status", metric)
                interface_operational_Status_data = zbx.get_metric_data(
                    hostname=hostname,
                    metric_name=interface_operational_Status,
                    time_from=time_from,
                    time_to=time_to,
                    statistical_measure='last'
                )
                if interface_operational_Status_data['data']:
                    operational_status_value = interface_operational_Status_data['data'][0]['value']
                    if operational_status_value is not None:
                        supporting_data[f'{interface_name} Operational Status'] = operational_status_value
                        
    return supporting_data

def get_time_series_data(
    hostname: str,
    metric_name: List[str],
    time_from: int,
    time_to: int,
    chunk_size: int = 24 * 3600,  # Process 1 day at a time
    
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:

    log_file = '/root/zabbix-lib/logs/resource_utilization.log'
    # Setup logger
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

    # Input validation
    if not metric_name:
        logger.error("metric_name list cannot be empty.")
        return None, None
    if time_from >= time_to:
        logger.error("time_from must be less than time_to.")
        return None, None
    if chunk_size <= 0:
        logger.error("chunk_size must be positive.")
        return None, None

    time_span_seconds = time_to - time_from
    bin_seconds = get_time_diff(time_span_seconds)


    # Initialize lists for binned data and statistics
    dfs = []
    stats_list = []

    # Process data in chunks
    current_time = time_from
    while current_time < time_to:
        chunk_end = min(current_time + chunk_size, time_to)
        chunk_dfs = []
        for metric in metric_name:
            # Fetch metric data for the chunk
            data = zbx.get_metric_data(
                hostname=hostname,
                metric_name=metric,
                time_from=current_time,
                time_to=chunk_end
            )

            if 'error' in data:
                logger.error(f"Error fetching data for {metric}: {data['error']}")
                return None, None

            data_list = data.get('data', [])
            if not data_list:
                logger.warning(f"No data returned for metric {metric} in chunk {current_time}-{chunk_end}")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data_list)

            if 'clock' not in df.columns or 'value' not in df.columns:
                logger.error(f"Missing 'clock' or 'value' column in data for {metric}")
                return None, None

            # Optimize data types
            try:
                df['clock'] = df['clock'].astype('int32')
                df['value'] = pd.to_numeric(df['value'], errors='coerce', downcast='float').round(2)
            except Exception as e:
                logger.error(f"Error converting data types for {metric}: {e}")
                return None, None

            # Compute statistics for this metric in this chunk
            if not df.empty:
                stats = pd.DataFrame({
                    'metric': [metric],
                    'min': [df['value'].min().round(2)],
                    'max': [df['value'].max().round(2)],
                    'mean': [df['value'].mean().round(2)]
                })
                stats_list.append(stats)

            # Floor clock to the bin
            df['clock_bin'] = (df['clock'] // bin_seconds) * bin_seconds

            # Aggregate by clock_bin (mean of values within each bin)
            df = df.groupby('clock_bin')['value'].mean().reset_index()
            df = df.rename(columns={'value': metric})

            chunk_dfs.append(df)

        if chunk_dfs:
            # Merge DataFrames in this chunk
            df_chunk = reduce(
                lambda left, right: pd.merge(left, right, on='clock_bin', how='inner'),
                chunk_dfs
            )
            dfs.append(df_chunk)

        current_time += chunk_size

    # Check if any data was collected
    if not dfs:
        logger.error("No valid data collected for any metrics.")
        return None, None

    # Concatenate all chunks for binned data
    df_merged = pd.concat(dfs, ignore_index=True)

    # Aggregate again if there are overlapping bins across chunks
    df_merged = df_merged.groupby('clock_bin').mean().reset_index()

    # Sort by clock_bin
    df_merged = df_merged.sort_values('clock_bin').reset_index(drop=True)

    # Handle missing values
    df_merged = df_merged.fillna(0)

    # Combine statistics from all chunks
    if stats_list:
        df_stats = pd.concat(stats_list, ignore_index=True)
        # Aggregate statistics across chunks (min of mins, max of maxes, mean of means)
        df_stats = df_stats.groupby('metric').agg({
            'min': 'min',
            'max': 'max',
            'mean': 'mean'
        }).reset_index()
    else:
        df_stats = pd.DataFrame(columns=['metric', 'min', 'max', 'mean'])

    return df_merged, df_stats

def generate_pdf_report(
    df,
    report_type,
    from_time,
    to_time,
    metadata=None,
    statistics=None,
    chart_path=None,
    severity_pie_chart=None,
    logo_path="assets/Quess.png",
    template_dir="templates",
    template_name="report_template.html"
):
    """
    Generates a PDF report using a pre-generated chart image.
    """
    try:
        # Load template
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)

        # Prepare logo path
        abs_logo_path = os.path.abspath(logo_path)
        if not os.path.exists(abs_logo_path):
            print(f"Logo not found: {abs_logo_path}")
        logo_uri = f"file://{abs_logo_path}"

        # Prepare chart URI if provided
        chart_paths = []
        if chart_path:
            abs_chart_path = os.path.abspath(chart_path)
            if not os.path.exists(abs_chart_path):
                print(f"Chart image not found: {abs_chart_path}")
            chart_paths.append({
                "metric": "Combined Metrics",
                "path": f"file://{abs_chart_path}"
            })

        span = duration_to_label(from_time, to_time)

        severity_pie_chart_uri = None
        if severity_pie_chart:
            abs_pie_path = os.path.abspath(severity_pie_chart)
            if os.path.exists(abs_pie_path):
                severity_pie_chart_uri = f"file://{abs_pie_path}"
            else:
                print(f"Severity pie chart not found: {abs_pie_path}")

        
        # Render HTML
        html_content = template.render(
            df=df,
            report_type=report_type,
            span=span,
            beginning=unixtotime(from_time),
            generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            logo_path=logo_uri,
            chart_paths=chart_paths,
            metadata=metadata or {},
            statistics=statistics,
            severity_pie_chart=severity_pie_chart_uri
        )

        # Prepare output filename
        safe_report_type = report_type.replace(" ", "_").replace("/", "_")
        output_file = f"{safe_report_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        pdf_path = os.path.join(os.getcwd(), output_file)

        # PDF options
        options = {
            'enable-local-file-access': ''
        }

        # Generate PDF
        pdfkit.from_string(html_content, pdf_path, options=options)
        print(f"PDF generated at: {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"PDF generation failed: {e}")
        return None

def plot_all_metrics(df, output_path="multi_series_plot.png", threshold=None, unit=None):
    """
    Plots all columns (except the first) as separate series over time (first column is UNIX time).
    Optionally plots a horizontal threshold line if threshold is provided.
    Formats y-axis based on the specified unit (bytes, percentage, or custom).
    Ensures equal spacing for y-axis ticks.
    The entire graph is saved with a transparent background.
    """
    try:
        # Extract time column
        time_col = df.columns[0]
        metrics = df.columns[1:]
        
        # Convert UNIX time to datetime
        df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
        x = df[time_col]
        
        # Choose a color palette
        colors = plt.cm.tab10.colors  # 10 distinct colors
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for idx, metric in enumerate(metrics):
            y = df[metric]
            ax.plot(
                x,
                y,
                label=metric,
                linewidth=2,
                color=colors[idx % len(colors)],
                zorder=3
            )
            ax.fill_between(
                x,
                y,
                color=colors[idx % len(colors)],
                alpha=0.15,
                zorder=1
            )
        
        # Plot threshold line if provided
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Threshold',
                zorder=2
            )
        
        # Axis labels
        ax.set_xlabel("Time")
        
        # Format y-axis based on unit
        if unit is not None:
            if unit.lower() == 'bytes':
                def format_bytes(value, _):
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if value < 1024:
                            return f"{value:.1f} {unit}"
                        value /= 1024
                    return f"{value:.1f} PB"
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
                ax.set_ylabel("Value (Bytes)")
            elif unit.lower() == 'percentage':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
                ax.set_ylabel("Value (%)")
            else:
                ax.set_ylabel(f"Value ({unit})")
        else:
            ax.set_ylabel("Value")
        
        # Ensure equal spacing for y-axis ticks
        y_max = df[metrics].max().max()  # Maximum value across all metrics
        y_min = 0  # Since ylim is set to start at 0
        if y_max > 0:
            # Calculate a reasonable step size for ticks (e.g., aim for ~5-10 ticks)
            num_ticks = 8
            tick_step = y_max / num_ticks
            # Round tick_step to a "nice" number
            tick_step = round(tick_step, -int(np.floor(np.log10(tick_step))) + 1)
            ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        
        # Custom legend with square markers
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker='s',
                color='w',
                markerfacecolor=colors[idx % len(colors)],
                markersize=10,
                linestyle='None',
                label=metric
            )
            for idx, metric in enumerate(metrics)
        ]
        
        # Add threshold to legend if it exists
        if threshold is not None:
            custom_legend.append(
                Line2D(
                    [0],
                    [0],
                    color='red',
                    linestyle='--',
                    linewidth=1.5,
                    label='Threshold'
                )
            )
        
        # Legend styling
        ax.legend(
            handles=custom_legend,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            frameon=False,
            fontsize=10,
            ncol=len(metrics) + (1 if threshold is not None else 0)
        )
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#888")
        ax.spines['bottom'].set_color("#888")
        
        # Transparent backgrounds
        ax.set_facecolor("none")
        fig.patch.set_facecolor("none")
        
        ax.set_xlim(left=x.min(), right=x.max())
        ax.set_ylim(bottom=0)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure with transparency
        plt.savefig(output_path, bbox_inches="tight", transparent=True)
        plt.close()
        
        print(f"Plot saved to: {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)
    
    except Exception as e:
        print(f"Failed to plot: {e}")
        return None

def main_f(time_from: int, time_to: int, hostname: Optional[List[str]] = None, metric_name: Optional[List[str]] = None, report_type = None):

    if report_type == 'interface':
        interface_metric = []
        for metric in metric_name:
            if metric.startswith('interface'):
                metric  = metric.replace('interface: ','')
            d = zbx.get_host_drives(hostname=hostname)
            d = pd.DataFrame(d['data'])
            d
            df = d[d['name'].str.contains(rf'Interface {metric}.*(?:Bits sent|Bits received)', regex=True)]
            item = df['name']
            for i in item:
                interface_metric.append(i)
                        
        # Get time series data
        t = get_time_series_data(
            hostname=hostname,
            metric_name=interface_metric,
            time_from=time_from,
            time_to=time_to
        )
        
        meta_data = get_host_support_data(
            hostname=hostname,
            metric_name=interface_metric,
            time_from=time_from,
            time_to=time_to)
        
        graph_path = plot_all_metrics(t[0],unit='bytes')

        for col in t[0].columns:
            if col != 'clock_bin':
                t[0][col] = t[0][col].apply(format_interface_bytes)
        for col in t[1].columns:
            if col != 'metric':
                t[1][col] = t[1][col].apply(format_interface_bytes)
        subject = f"Interface Report for {hostname}"

    elif report_type == 'filesystem':
        drive_metric = []
        if metric_name[0] == 'all':
            d = zbx.get_host_drives(hostname=hostname)
            d = pd.DataFrame(d['data'])
            df = d[d['name'].str.contains(r'^FS \[.*\]: Space: Used, in %$', regex=True)]
            item = df['name']
            for i in item:
                drive_metric.append(i)
        else:
            for metric in metric_name:
                metric = metric.replace('filesystem: ','')
                d = zbx.get_host_drives(hostname=hostname)
                d = pd.DataFrame(d['data'])
                df = d[d['name'].str.contains(f'FS [{metric_name}]: Space: Used in %')]
                item = df['name']
                for i in item:
                    drive_metric.append(i)
        metric_name = drive_metric

        # Get time series data
        t = get_time_series_data(
            hostname=hostname,
            metric_name=metric_name,
            time_from=time_from,
            time_to=time_to
        )
        
        meta_data = get_host_support_data(
            hostname=hostname,
            metric_name=metric_name,
            time_from=time_from,
            time_to=time_to)
        
        graph_path = plot_all_metrics(t[0])

        subject = f"Drive Utilization Report for {hostname}"

    elif report_type == 'alert':
        result = zbx.get_alerts(
        hostname=hostname,
        time_from=time_from,
        time_to=time_to,)

        df = pd.DataFrame(result)

        df['Status'] = df['recovery_eventid'].isna().astype(int)



        for col in ['start_time', 'end_time']:
            df[col] = df[col].apply(unixtotime)

        df['duration'] = df['duration'].apply(seconds_to_compact_time)
        # 3. Drop unwanted columns
        df = df.drop(columns=['trigger_name', 'eventid', 'recovery_eventid'])

        def generate_severity_combined_chart(df, output_path='severity_combined_chart.png'):

            # Fixed severity order and colors
            severity_order = ['Disaster', 'High', 'Average', 'Warning', 'Information', 'Not classified']
            severity_color_map = {
                'Disaster': '#e74c3c',
                'High': '#e67e22',
                'Average': '#f39c12',
                'Warning': '#f1c40f',
                'Information': '#3498db',
                'Not classified': '#95a5a6'
            }

            # Count severity levels, ensure all severities are present
            severity_counts = df['severity'].value_counts().reindex(severity_order, fill_value=0)

            # Filter out 0-count entries
            filtered = severity_counts[severity_counts > 0]
            labels = filtered.index.tolist()
            values = filtered.values.tolist()

            # Map colors consistently
            colors = [severity_color_map[label] for label in labels]
            explode = [0.1 if count == max(values) and count > 0 else 0 for count in values]

            # Create side-by-side plots
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='#f9f9f9')

            # Pie chart (left)
            axes[0].pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                explode=explode,
                shadow=True,
                labeldistance=1.15,  # Push labels outward
                pctdistance=0.75     # Bring percentages inward
            )
            axes[0].set_title("Alert Severity Distribution", fontsize=11)

            # Bar chart (right)
            bars = axes[1].bar(labels, values, color=colors)
            axes[1].set_title("Severity Count", fontsize=11)
            axes[1].set_ylabel("Count")
            axes[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

            # Add values above bars
            for bar in bars:
                height = bar.get_height()
                axes[1].annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path, facecolor=fig.get_facecolor())
            plt.close()
            return output_path
        
        # Generate pie chart and include in PDF
        combined_chart_path = generate_severity_combined_chart(df, 'severity_combined_chart.png')

        subject = "Alert Summary Report"
        pdf_path = generate_pdf_report(
        df=df,
        report_type=subject,
        from_time=time_from,
        to_time=time_to,
        severity_pie_chart=combined_chart_path
        )

        email_response = email_sender.send_email_with_attachment(
            to_email=email_config.get("to_email",[]),
            subject=subject,
            body=f"Please find the attached report for {hostname} covering the period from {unixtotime(time_from)} to {unixtotime(time_to)}.",
            file_name=pdf_path
        )
        
        return df
    else:
        # Get time series data
        t = get_time_series_data(
            hostname=hostname,
            metric_name=metric_name,
            time_from=time_from,
            time_to=time_to
        )
        
        meta_data = get_host_support_data(
            hostname=hostname,
            metric_name=metric_name,
            time_from=time_from,
            time_to=time_to)
        
        graph_path = plot_all_metrics(t[0])

        subject = f"Resource utilization Report for {hostname}"

    pdf_path = generate_pdf_report(
        df=t[0],
        report_type=subject,
        from_time=time_from,
        to_time=time_to,
        metadata=meta_data,
        statistics=t[1],
        chart_path=graph_path
    )

    email_response = email_sender.send_email_with_attachment(
        to_email=email_config.get("to_email",[]),
        subject=subject,
        body=f"Please find the attached report for {hostname} covering the period from {unixtotime(time_from)} to {unixtotime(time_to)}.",
        file_name=pdf_path
    )
    
    return t[0], t[1], meta_data, email_response
