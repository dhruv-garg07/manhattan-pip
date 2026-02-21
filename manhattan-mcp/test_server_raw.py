
import subprocess
import json
import sys
import time
import os

def test_server_raw():
    # Construct the tool call request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "index_file",
            "arguments": {
                "file_path": r"c:\Desktop\python_workspace_311\PdM-main\PdM-main\manhattan-pip\manhattan-mcp\src\manhattan_mcp\config.py",
                "agent_id": "test_agent"
            }
        }
    }
    
    request_str = json.dumps(request) + "\n"
    print(f"Sending request: {request_str.strip()}")
    
    # Run the server as a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "manhattan_mcp.cli", "start"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd()
    )
    
    # Send the request
    try:
        stdout_data, stderr_data = process.communicate(input=request_str, timeout=30)
        
        print("\n--- RAW STDOUT ---")
        print(stdout_data)
        print("--- END RAW STDOUT ---\n")
        
        print("\n--- RAW STDERR ---")
        print(stderr_data)
        print("--- END RAW STDERR ---\n")
        
        if stdout_data:
            # Check for the first non-whitespace character
            stripped = stdout_data.strip()
            if stripped:
                print(f"First character of stdout: '{stripped[0]}'")
                if stripped[0] != '{':
                    print("ERROR: Stdout does not start with '{'!")
        
    except subprocess.TimeoutExpired:
        process.kill()
        print("Timed out waiting for server response.")

if __name__ == "__main__":
    # Add src to path for the subprocess
    os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    test_server_raw()
