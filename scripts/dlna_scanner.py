#!/usr/bin/env python3
"""
DLNA/UPnP Media Server Scanner

Discovers DLNA servers on the network and browses their content.
Useful for debugging minidlna and other media servers.

Usage:
    python dlna_scanner.py              # Discover servers and list content
    python dlna_scanner.py --search TV  # Search for files containing "TV"
    python dlna_scanner.py --deep       # Deep scan all folders
"""

import argparse
import socket
import struct
import sys
import time
from typing import Optional
from xml.etree import ElementTree

try:
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

console = Console() if RICH_AVAILABLE else None

# SSDP Discovery
SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
SSDP_MX = 3
SSDP_ST = "urn:schemas-upnp-org:device:MediaServer:1"

# UPnP namespaces
NS = {
    'dc': 'http://purl.org/dc/elements/1.1/',
    'upnp': 'urn:schemas-upnp-org:metadata-1-0/upnp/',
    'didl': 'urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/',
    'dlna': 'urn:schemas-dlna-org:metadata-1-0/',
}


def ssdp_discover(timeout: int = 5) -> list:
    """Discover DLNA media servers using SSDP."""

    ssdp_request = (
        f"M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n"
        f"MAN: \"ssdp:discover\"\r\n"
        f"MX: {SSDP_MX}\r\n"
        f"ST: {SSDP_ST}\r\n"
        f"\r\n"
    )

    servers = []
    seen_locations = set()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)

    # Send discovery request
    sock.sendto(ssdp_request.encode(), (SSDP_ADDR, SSDP_PORT))

    if console:
        console.print("[cyan]Discovering DLNA servers...[/cyan]")
    else:
        print("Discovering DLNA servers...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            data, addr = sock.recvfrom(4096)
            response = data.decode('utf-8', errors='ignore')

            # Parse response headers
            headers = {}
            for line in response.split('\r\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.upper().strip()] = value.strip()

            location = headers.get('LOCATION', '')
            if location and location not in seen_locations:
                seen_locations.add(location)
                server_info = get_server_info(location)
                if server_info:
                    server_info['address'] = addr[0]
                    servers.append(server_info)

        except socket.timeout:
            break
        except Exception as e:
            continue

    sock.close()
    return servers


def get_server_info(location: str) -> Optional[dict]:
    """Get server information from device description."""
    try:
        response = requests.get(location, timeout=5)
        root = ElementTree.fromstring(response.content)

        # Find device element (handle namespace)
        ns = {'d': 'urn:schemas-upnp-org:device-1-0'}
        device = root.find('.//d:device', ns)

        if device is None:
            # Try without namespace
            device = root.find('.//device')

        if device is None:
            return None

        def get_text(elem, tag):
            child = elem.find(f'd:{tag}', ns)
            if child is None:
                child = elem.find(tag)
            return child.text if child is not None else ''

        # Find ContentDirectory service
        content_dir_url = None
        services = device.findall('.//d:service', ns)
        if not services:
            services = device.findall('.//service')

        for service in services:
            service_type = get_text(service, 'serviceType')
            if 'ContentDirectory' in service_type:
                control_url = get_text(service, 'controlURL')
                # Build full URL
                from urllib.parse import urljoin
                content_dir_url = urljoin(location, control_url)
                break

        return {
            'name': get_text(device, 'friendlyName'),
            'model': get_text(device, 'modelName'),
            'manufacturer': get_text(device, 'manufacturer'),
            'location': location,
            'content_dir_url': content_dir_url,
        }

    except Exception as e:
        return None


def browse_content(control_url: str, object_id: str = '0', flag: str = 'BrowseDirectChildren') -> list:
    """Browse content directory."""

    soap_body = f'''<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
    <s:Body>
        <u:Browse xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">
            <ObjectID>{object_id}</ObjectID>
            <BrowseFlag>{flag}</BrowseFlag>
            <Filter>*</Filter>
            <StartingIndex>0</StartingIndex>
            <RequestedCount>1000</RequestedCount>
            <SortCriteria></SortCriteria>
        </u:Browse>
    </s:Body>
</s:Envelope>'''

    headers = {
        'Content-Type': 'text/xml; charset="utf-8"',
        'SOAPACTION': '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
    }

    try:
        response = requests.post(control_url, data=soap_body, headers=headers, timeout=10)
        root = ElementTree.fromstring(response.content)

        # Find Result element
        result_elem = root.find('.//{urn:schemas-upnp-org:service:ContentDirectory:1}Result')
        if result_elem is None:
            result_elem = root.find('.//Result')

        if result_elem is None or not result_elem.text:
            return []

        # Parse DIDL-Lite result
        didl_root = ElementTree.fromstring(result_elem.text)

        items = []

        # Parse containers (folders)
        for container in didl_root.findall('.//didl:container', NS):
            items.append({
                'type': 'container',
                'id': container.get('id'),
                'title': container.findtext('dc:title', '', NS),
                'child_count': container.get('childCount', '?'),
            })

        # Try without namespace prefix
        for container in didl_root.findall('.//{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}container'):
            title_elem = container.find('.//{http://purl.org/dc/elements/1.1/}title')
            items.append({
                'type': 'container',
                'id': container.get('id'),
                'title': title_elem.text if title_elem is not None else 'Unknown',
                'child_count': container.get('childCount', '?'),
            })

        # Parse items (files)
        for item in didl_root.findall('.//didl:item', NS):
            title_elem = item.find('dc:title', NS)
            upnp_class = item.findtext('upnp:class', '', NS)
            res_elem = item.find('didl:res', NS)

            items.append({
                'type': 'item',
                'id': item.get('id'),
                'title': title_elem.text if title_elem is not None else 'Unknown',
                'class': upnp_class,
                'url': res_elem.text if res_elem is not None else '',
                'size': res_elem.get('size', '') if res_elem is not None else '',
            })

        # Try without namespace prefix
        for item in didl_root.findall('.//{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}item'):
            title_elem = item.find('.//{http://purl.org/dc/elements/1.1/}title')
            class_elem = item.find('.//{urn:schemas-upnp-org:metadata-1-0/upnp/}class')
            res_elem = item.find('.//{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}res')

            items.append({
                'type': 'item',
                'id': item.get('id'),
                'title': title_elem.text if title_elem is not None else 'Unknown',
                'class': class_elem.text if class_elem is not None else '',
                'url': res_elem.text if res_elem is not None else '',
                'size': res_elem.get('size', '') if res_elem is not None else '',
            })

        return items

    except Exception as e:
        if console:
            console.print(f"[red]Error browsing content:[/red] {e}")
        else:
            print(f"Error browsing content: {e}")
        return []


def format_size(size_str: str) -> str:
    """Format size string to human readable."""
    if not size_str:
        return ''
    try:
        size = int(size_str)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except:
        return size_str


def display_servers(servers: list):
    """Display discovered servers."""
    if not servers:
        if console:
            console.print("[yellow]No DLNA servers found.[/yellow]")
        else:
            print("No DLNA servers found.")
        return

    if RICH_AVAILABLE:
        table = Table(title="Discovered DLNA Servers")
        table.add_column("#", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Model", style="white")
        table.add_column("Address", style="yellow")

        for i, server in enumerate(servers, 1):
            table.add_row(
                str(i),
                server.get('name', 'Unknown'),
                server.get('model', ''),
                server.get('address', ''),
            )

        console.print(table)
    else:
        print("\nDiscovered DLNA Servers:")
        for i, server in enumerate(servers, 1):
            print(f"  {i}. {server.get('name', 'Unknown')} ({server.get('address', '')})")


def browse_recursive(control_url: str, object_id: str = '0', depth: int = 0, max_depth: int = 3, search_term: str = None) -> list:
    """Recursively browse content."""
    all_items = []
    items = browse_content(control_url, object_id)

    for item in items:
        item['depth'] = depth

        if search_term:
            if search_term.lower() in item.get('title', '').lower():
                all_items.append(item)
        else:
            all_items.append(item)

        # Recurse into containers
        if item['type'] == 'container' and depth < max_depth:
            children = browse_recursive(control_url, item['id'], depth + 1, max_depth, search_term)
            all_items.extend(children)

    return all_items


def display_content(items: list, title: str = "Content"):
    """Display content items."""
    if not items:
        if console:
            console.print("[yellow]No content found.[/yellow]")
        else:
            print("No content found.")
        return

    if RICH_AVAILABLE:
        # Build tree
        tree = Tree(f"[bold]{title}[/bold]")

        containers = {}
        root_items = []

        for item in items:
            depth = item.get('depth', 0)
            if item['type'] == 'container':
                node_text = f"[blue]{item['title']}[/blue] ({item.get('child_count', '?')} items)"
            else:
                size = format_size(item.get('size', ''))
                size_text = f" [{size}]" if size else ""
                node_text = f"[green]{item['title']}[/green]{size_text}"

            if depth == 0:
                branch = tree.add(node_text)
                if item['type'] == 'container':
                    containers[item['id']] = branch
            else:
                # For simplicity, add to root
                tree.add("  " * depth + node_text)

        console.print(tree)

        # Summary
        video_count = sum(1 for i in items if i['type'] == 'item' and 'video' in i.get('class', '').lower())
        audio_count = sum(1 for i in items if i['type'] == 'item' and 'audio' in i.get('class', '').lower())
        folder_count = sum(1 for i in items if i['type'] == 'container')

        console.print(f"\n[cyan]Summary:[/cyan] {folder_count} folders, {video_count} videos, {audio_count} audio files")

    else:
        print(f"\n{title}:")
        for item in items:
            indent = "  " * item.get('depth', 0)
            if item['type'] == 'container':
                print(f"{indent}[DIR] {item['title']} ({item.get('child_count', '?')} items)")
            else:
                size = format_size(item.get('size', ''))
                print(f"{indent}      {item['title']} {size}")


def main():
    parser = argparse.ArgumentParser(description='DLNA Media Server Scanner')
    parser.add_argument('--search', '-s', help='Search for files containing this term')
    parser.add_argument('--deep', '-d', action='store_true', help='Deep scan all folders')
    parser.add_argument('--timeout', '-t', type=int, default=5, help='Discovery timeout (default: 5)')
    parser.add_argument('--server', type=int, help='Select server by number')

    args = parser.parse_args()

    # Discover servers
    servers = ssdp_discover(args.timeout)
    display_servers(servers)

    if not servers:
        sys.exit(1)

    # Select server
    if args.server:
        server_idx = args.server - 1
    else:
        server_idx = 0

    if server_idx >= len(servers):
        if console:
            console.print(f"[red]Invalid server number[/red]")
        else:
            print("Invalid server number")
        sys.exit(1)

    server = servers[server_idx]

    if console:
        console.print(f"\n[bold]Browsing:[/bold] {server['name']}")
    else:
        print(f"\nBrowsing: {server['name']}")

    if not server.get('content_dir_url'):
        if console:
            console.print("[red]Could not find ContentDirectory service[/red]")
        else:
            print("Could not find ContentDirectory service")
        sys.exit(1)

    # Browse content
    max_depth = 5 if args.deep else 2
    items = browse_recursive(
        server['content_dir_url'],
        '0',
        max_depth=max_depth,
        search_term=args.search
    )

    title = f"Search results for '{args.search}'" if args.search else "Media Library"
    display_content(items, title)


if __name__ == '__main__':
    main()
