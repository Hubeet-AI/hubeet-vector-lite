import socket, time
s = socket.socket()
s.connect(("127.0.0.1", 5555))
s.sendall(b"TSET xxx \"test\"")
s.recv(1024)
start = time.time()
s.sendall(b"TDEL xxx")
s.recv(1024)
print("TDEL time:", time.time() - start)
