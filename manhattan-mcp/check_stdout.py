
import subprocess
import sys
import os
import json
import time

def check_tool_stdout():
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
    
    # Run server
    process = subprocess.Popen(
        [sys.executable, "-m", "manhattan_mcp.cli", "start"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    # Send request and capture output
    try:
        stdout_data, stderr_data = process.communicate(input=request_str.encode('utf-8'), timeout=30)
        
        print(f"Stdout length: {len(stdout_data)}")
        if len(stdout_data) > 0:
            # Look for non-JSON characters at the start
            first_char = stdout_data[0:1].decode('utf-8', errors='ignore')
            print(f"First character of stdout: '{first_char}' (hex: {stdout_data[0:1].hex()})")
            
            if not first_char.startswith('{'):
                print("--- START OF STDOUT ---")
                print(stdout_data[:100].decode('utf-8', errors='ignore'))
                print("--- END ---")
        else:
            print("Stdout is EMPTY.")

        print(f"Stderr length: {len(stderr_data)}")
        if len(stderr_data) > 0:
            print("--- START OF STDERR ---")
            print(stderr_data[:500].decode('utf-8', errors='ignore'))
            print("--- END ---")
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("Timed out.")

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    check_tool_stdout()
