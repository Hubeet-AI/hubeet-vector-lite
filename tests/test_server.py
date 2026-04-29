import socket
import time

def test_server():
    dim = 384
    port = 5555
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", port))
    
    print("Testing PING...")
    s.sendall(b"PING\r\n")
    print("Response:", s.recv(1024).decode())
    
    print("Testing VSET...")
    vec_data = " ".join(["0.5"] * dim)
    cmd = f"VSET vec_test {vec_data}\r\n"
    s.sendall(cmd.encode())
    print("Response:", s.recv(1024).decode())
    
    print("Testing VSEARCH...")
    query_data = " ".join(["0.6"] * dim)
    cmd = f"VSEARCH 1 {query_data}\r\n"
    s.sendall(cmd.encode())
    print("Response:", s.recv(1024).decode())
    
    print("Testing QUIT...")
    s.sendall(b"QUIT\r\n")
    s.close()

if __name__ == "__main__":
    test_server()
