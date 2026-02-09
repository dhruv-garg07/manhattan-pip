
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from manhattan_mcp.server import get_data_dir, api
    
    print(f"✅ Import successful")
    
    data_dir = get_data_dir()
    print(f"📂 Resolved Data Directory: {data_dir}")
    
    expected_part = "Library/Application Support/manhattan-mcp/data"
    if expected_part in str(data_dir):
        print("✅ Path looks correct for macOS")
    else:
        print(f"⚠️ Path might be unexpected for macOS (Expected to contain: {expected_part})")

    if data_dir.exists():
        print("✅ Data directory exists (created successfully)")
    else:
        print("❌ Data directory was NOT created")
        
    print(f"✅ LocalAPI initialized with root_path: {api.root_path}")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
