import socket
import time
import subprocess
import os
import signal

def send_cmd(s, cmd):
    s.sendall(cmd.encode() + b"\r\n")
    return s.recv(1024).decode()

def test_persistence():
    dim = 384
    port = 5555
    vec_data = " ".join(["0.7"] * dim)
    
    # Kill any existing server
    subprocess.run(["pkill", "-9", "hvl-server"], stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    # Clean up old dump
    if os.path.exists("dump.hvl"):
        os.remove("dump.hvl")
        
    print("Step 1: Start server and insert data")
    srv = subprocess.Popen(["./hvl-server"])
    time.sleep(1)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", port))
    
    print("VSET response:", send_cmd(s, f"VSET persist_test {vec_data}"))
    print("SAVE response:", send_cmd(s, "SAVE"))
    time.sleep(0.5)
    
    s.close()
    srv.kill()
    srv.wait()
    print("Server stopped.")
    
    if os.path.exists("dump.hvl"):
        print(f"dump.hvl size: {os.path.getsize('dump.hvl')} bytes")
    else:
        print("dump.hvl NOT FOUND!")
    
    print("\nStep 2: Restart server and verify data")
    srv = subprocess.Popen(["./hvl-server"])
    time.sleep(1)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", port))
    
    query_data = " ".join(["0.71"] * dim)
    print("VSEARCH response:", send_cmd(s, f"VSEARCH 1 {query_data}"))
    
    s.close()
    srv.kill()
    srv.wait()
    print("Verification complete.")

if __name__ == "__main__":
    test_persistence()
