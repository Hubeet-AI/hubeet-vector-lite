import socket
import os
import time

class HVLProtocolParser:
    def __init__(self, sock):
        self.reader = sock.makefile('rb')

    def read_response(self):
        line = self.reader.readline()
        if not line: return None
        line = line.decode().strip()
        if not line: return self.read_response()
        
        prefix = line[0]
        payload = line[1:]
        
        if prefix == '+': return payload
        elif prefix == '-': return f"Error: {payload}"
        elif prefix == ':': return int(payload)
        elif prefix == '$':
            length = int(payload)
            if length == -1: return None
            data = self.reader.read(length)
            self.reader.readline() # Consume \r\n
            return data.decode()
        elif prefix == '*':
            count = int(payload)
            if count == -1: return None
            return [self.read_response() for _ in range(count)]
        return line

def test_upsert_delete():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 5555))
    parser = HVLProtocolParser(s)
    
    print("--- UPSERT TEST ---")
    
    # 1. Insert
    s.sendall(b"TSET upsert_test \"The quick brown fox jumps over the lazy dog\"\r\n")
    print(f"Insert 1: {parser.read_response()}")
    
    # 2. Search (Old)
    s.sendall(b"TSEARCH 1 \"fox jumps\"\r\n")
    res1 = parser.read_response()
    print(f"Search 1 Result: {res1}")
    
    # 3. Update (UPSERT)
    print("\nApplying UPSERT (same ID, different text)...")
    s.sendall(b"TSET upsert_test \"Blue electric cars are efficient in cities\"\r\n")
    print(f"Update response: {parser.read_response()}")
    
    # 4. Search (New content)
    s.sendall(b"TSEARCH 1 \"electric cars\"\r\n")
    res2 = parser.read_response()
    print(f"Search 2 (New Content) Result: {res2}")
    
    # 5. Search (Old content - should be weaker or different)
    s.sendall(b"TSEARCH 1 \"fox jumps\"\r\n")
    res3 = parser.read_response()
    print(f"Search 3 (Old Content) Result: {res3}")
    
    # 6. Delete
    print("\nDeleting ID...")
    s.sendall(b"DELETE upsert_test\r\n")
    print(f"Delete response: {parser.read_response()}")
    
    # 7. Final search
    s.sendall(b"TSEARCH 1 \"electric cars\"\r\n")
    res4 = parser.read_response()
    print(f"Final Search Result: {res4}")
    
    s.close()

if __name__ == "__main__":
    test_upsert_delete()
