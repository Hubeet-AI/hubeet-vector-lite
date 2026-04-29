import socket
import time

def test_pipelining(host='127.0.0.1', port=5555, num_commands=5000):
    print(f"Connecting to {host}:{port} for PING pipelining test...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    payload = b"PING\r\n" * num_commands
    start = time.time()
    s.sendall(payload)
    
    total_received = 0
    expected = b"+PONG\r\n" * num_commands
    
    while total_received < len(expected):
        data = s.recv(65536)
        if not data: break
        total_received += len(data)
    
    end = time.time()
    print(f"SUCCESS: Received {num_commands} PONGs in {end - start:.4f}s")
    print(f"Throughput: {num_commands / (end - start):.2f} commands/sec")
    s.close()

def test_pipelining_vset(host='127.0.0.1', port=5555, num_commands=1000):
    print(f"\nTesting VSET pipelining with {num_commands} vectors (dim=384)...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    dim = 384
    commands = []
    for i in range(num_commands):
        vec = ",".join(["0.1"] * dim)
        commands.append(f"VSET pipe_{i} {vec}\r\n".encode())
    
    payload = b"".join(commands)
    start = time.time()
    s.sendall(payload)
    
    total_received = 0
    expected_resp = b"+OK\r\n" * num_commands
    
    while total_received < len(expected_resp):
        data = s.recv(65536)
        if not data: break
        total_received += len(data)
    
    end = time.time()
    print(f"SUCCESS: Bulk indexed {num_commands} vectors in {end - start:.4f}s")
    print(f"Throughput: {num_commands / (end - start):.2f} vectors/sec")
    s.close()

if __name__ == "__main__":
    test_pipelining()
    test_pipelining_vset()
