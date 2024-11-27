from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List, Optional, Dict, Set
from pathlib import Path
import subprocess
import asyncio
import signal
import os
import socket
from datetime import datetime


class PortManager:
    def __init__(self, min_port: int = 8501):
        self.min_port = min_port
        self.used_ports: Set[int] = set()

    def is_port_available(self, port: int) -> bool:
        """Check if a port is available on localhost"""
        if port in self.used_ports:
            return False

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return True
            except OSError:
                return False

    def get_next_available_port(self, preferred_port: Optional[int] = None) -> int:
        """Get next available port, starting from preferred_port or min_port"""
        start_port = max(preferred_port or self.min_port, self.min_port)

        current_port = start_port
        while current_port < 65535:
            if self.is_port_available(current_port):
                self.used_ports.add(current_port)
                return current_port
            current_port += 1
        raise RuntimeError("No available ports found")

    def release_port(self, port: int):
        """Release a port from used_ports set"""
        self.used_ports.discard(port)


class StreamlitProcess:
    def __init__(
        self, demo_name: str, script_path: str, port: int, host: str = "localhost"
    ):
        self.demo_name = demo_name
        self.script_path = script_path
        self.port = port
        self.host = host
        self.process: Optional[subprocess.Popen] = None
        self.url = f"http://{host}:{port}"

    async def start(self):
        cmd = [
            "streamlit",
            "run",
            str(self.script_path),
            "--server.port",
            str(self.port),
            "--server.headless",
            "true",
            "--server.address",
            self.host,
        ]

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(
            f"Started Streamlit process for {self.script_path} on {self.host}:{self.port}"
        )

    async def stop(self):
        if self.process:
            if os.name == "nt":  # Windows
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)])
            else:  # Unix-like
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.create_subprocess_exec(
                                *["kill", "-TERM", f"{self.process.pid}"]
                            )
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            print(f"Stopped Streamlit process on {self.host}:{self.port}")


class DemoManager:
    def __init__(self):
        self.streamlit_processes: List[StreamlitProcess] = []
        self.config_data = {}
        self.port_manager = PortManager(min_port=8501)

    def get_demo_port(self, demo_name: str, demo_config: Dict) -> int:
        """Get port for demo from config or find a free port"""
        preferred_port = demo_config.get("port")

        if preferred_port is not None:
            if self.port_manager.is_port_available(preferred_port):
                allocated_port = preferred_port
            else:
                print(
                    f"Port {preferred_port} for {demo_name} is already in use. Finding another port..."
                )
                allocated_port = self.port_manager.get_next_available_port()
        else:
            allocated_port = self.port_manager.get_next_available_port()

        self.port_manager.used_ports.add(allocated_port)
        return allocated_port

    def get_demo_host(self, demo_config: Dict) -> str:
        """Get host for demo from config or use default"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            default_host = s.getsockname()[0]

        return demo_config.get("host", default_host)

    def generate_index_html(self) -> str:
        """Generate HTML index page with links to all running Streamlit apps"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RHEL AI Pro Suite Demo Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #262730;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .app-container {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .app-link {{
                    display: block;
                    color: #1E88E5;
                    text-decoration: none;
                    font-size: 18px;
                    margin-bottom: 10px;
                }}
                .app-link:hover {{
                    text-decoration: underline;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 14px;
                    margin-top: 20px;
                }}
                .status {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 14px;
                    margin-left: 10px;
                }}
                .status.running {{
                    background-color: #4CAF50;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RHEL AI Pro Suite Demo Dashboard</h1>
            </div>
            <div class="app-container">
        """

        for process in self.streamlit_processes:
            html_content += f"""
                <div>
                    <a href="{process.url}" class="app-link" target="_blank">
                        {process.demo_name}
                    </a>
                    <span class="status running">Running on {process.host}:{process.port}</span>
                </div>
            """

        html_content += f"""
                <div class="timestamp">
                    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        return html_content

    def load_demo_configs(self, config_data: Dict):
        """Load demo settings from provided configs"""
        self.config_data = config_data
        demos = self.config_data.get("demo", {})

        for demo_name, demo_config in demos.items():
            if not isinstance(demo_config, dict) or not demo_config.get(
                "enabled", False
            ):
                continue

            script_path = Path(f"./demo/{demo_name}/app.py")
            if not script_path.exists():
                print(f"Script not found for demo: {demo_name}")
                continue

            host = self.get_demo_host(demo_config)
            port = self.get_demo_port(demo_name, demo_config)

            self.streamlit_processes.append(
                StreamlitProcess(demo_name, script_path, port, host)
            )
            print(f"Configured {demo_name} to run on {host}:{port}")

    def setup_routes(self, app: FastAPI):
        """Set up FastAPI routes"""

        @app.get("/demo", response_class=HTMLResponse)
        async def read_root():
            return self.generate_index_html()

    async def startup(self, app: FastAPI, config_data: Dict):
        """Start all services with provided configs"""
        print("Starting up services...")

        # Load configurations and set up routes
        self.load_demo_configs(config_data)
        self.setup_routes(app)

        # Start Streamlit processes
        for process in self.streamlit_processes:
            print(
                f"Starting Streamlit process for {process.demo_name} on {process.host}:{process.port}"
            )
            await process.start()
            # Wait for startup
            await asyncio.sleep(5)
            print(f"Streamlit process started for {process.demo_name}")

    async def shutdown(self):
        """Stop all services"""
        for process in self.streamlit_processes:
            await process.stop()
            self.port_manager.release_port(process.port)
        print("All Streamlit processes stopped")
