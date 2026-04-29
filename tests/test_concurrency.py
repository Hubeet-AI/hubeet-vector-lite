import socket
import threading
import time
import random
import struct

def send_vset(port, vec_id, data):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", port))
        # Protocol: VSET <id> <dim> <data...>
        # For simplicity, we'll send a raw string that matches what our parser expects
        # VSET id dim float1 float2 ...
        dim = len(data)
        cmd = f"VSET {vec_id} {dim} " + " ".join(map(str, data)) + "\n"
        s.sendall(cmd.encode())
        res = s.recv(1024).decode()
        s.close()
        return res
    except Exception as e:
        return str(e)

def send_vsearch(port, data, k=5):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", port))
        dim = len(data)
        cmd = f"VSEARCH {k} {dim} " + " ".join(map(str, data)) + "\n"
        s.sendall(cmd.encode())
        res = s.recv(4096).decode()
        s.close()
        return res
    except Exception as e:
        return str(e)

def client_worker(worker_id, port, num_ops, stats):
    success = 0
    fail = 0
    for i in range(num_ops):
        v_id = f"worker_{worker_id}_vec_{i}"
        data = [random.random() for _ in range(384)]
        
        # 30% writers, 70% readers
        res = ""
        if random.random() < 0.3:
            res = send_vset(port, v_id, data)
        else:
            res = send_vsearch(port, data, k=3)
        
        if "+OK" in res or "*" in res:
            success += 1
        else:
            fail += 1
    stats[worker_id] = (success, fail)

def run_stress_test():
    NUM_WORKERS = 10
    OPS_PER_WORKER = 50
    PORT = 5555
    
    print(f"Starting stress test with {NUM_WORKERS} workers...")
    threads = []
    stats = {}
    start_time = time.time()
    
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=client_worker, args=(i, PORT, OPS_PER_WORKER, stats))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    end_time = time.time()
    total_success = sum(s[0] for s in stats.values())
    total_fail = sum(s[1] for s in stats.values())
    
    print(f"Stress test completed in {end_time - start_time:.2f} seconds.")
    print(f"Total Success: {total_success}, Total Fail: {total_fail}")
    
    if total_fail > 0:
        print("Test FAILED with some errors.")
        exit(1)
    else:
        print("Test PASSED.")

if __name__ == "__main__":
    run_stress_test()
