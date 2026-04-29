import socket
import os
import time

class HVLProtocolParser:
    def __init__(self, sock):
        self.reader = sock.makefile('rb')

    def read_response(self):
        line = self.reader.readline()
        if not line:
            return None
        line = line.decode().strip()
        
        prefix = line[0]
        payload = line[1:]
        
        if prefix == '+': # Status
            return payload
        elif prefix == '-': # Error
            return f"Error: {payload}"
        elif prefix == ':': # Integer
            return int(payload)
        elif prefix == '$': # Bulk String
            length = int(payload)
            if length == -1: return None
            data = self.reader.read(length)
            self.reader.readline() # Consume trailing \r\n
            return data.decode()
        elif prefix == '*': # Multi-bulk
            count = int(payload)
            if count == -1: return None
            return [self.read_response() for _ in range(count)]
        return line

def test_retrieval():
    host = "127.0.0.1"
    port = 5555
    docs_dir = "tests/test_docs"
    
    print(f"--- Hubeet Retrieval Evaluation ---")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        parser = HVLProtocolParser(s)
        
        # 1. Index Documents
        print("\n[Phase 1] Indexing documents...")
        for filename in sorted(os.listdir(docs_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(docs_dir, filename)
                with open(filepath, "r") as f:
                    text = f.read().replace("\n", " ").strip()
                    doc_id = filename
                    print(f"  Indexing: {doc_id}...", end=" ", flush=True)
                    s.sendall(f"TSET {doc_id} \"{text}\"\r\n".encode())
                    res = parser.read_response()
                    print(f"{res}")
        
        print("\nWaiting for graph stabilization...")
        time.sleep(2)
        
        # 2. Perform Semantic Searches
        queries = [
            ("How do plants get energy?", "photosynthesis.txt"),
            ("Explain deep learning and neural networks", "ai_evolution.txt"),
            ("What is a blockchain and bitcoin?", "crypto_finance.txt"),
            ("Tell me about the Roman Empire and Latin", "roman_empire.txt"),
            ("How do qubits work in computers?", "quantum_computing.txt"),
            ("biochemical reaction in green plants", "photosynthesis.txt"),
            ("decentralized finance and cryptographic assets", "crypto_finance.txt")
        ]
        
        print("\n[Phase 2] Evaluating retrieval quality...")
        total_tests = len(queries)
        passed_tests = 0
        
        for query, expected_id in queries:
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected_id}")
            
            # Send TSEARCH
            s.sendall(f"TSEARCH 3 \"{query}\"\r\n".encode())
            results = parser.read_response()
            
            if not isinstance(results, list):
                print(f"  Error: {results}")
                continue
            
            print(f"  Results (top {len(results)}):")
            
            found_correct = False
            for i, res in enumerate(results):
                # Results come as list of [id, dist]
                res_id = res[0]
                dist = res[1]
                
                rank = i + 1
                match_marker = "[MATCH]" if res_id == expected_id else ""
                print(f"    {rank}. {res_id} (dist: {dist}) {match_marker}")
                if res_id == expected_id and rank == 1:
                    found_correct = True
            
            if found_correct:
                print("  STATUS: SUCCESS (Top 1)")
                passed_tests += 1
            else:
                # Check if it's in top 3
                if any(r[0] == expected_id for r in results):
                    print("  STATUS: PARTIAL (In Top 3)")
                else:
                    print("  STATUS: FAILURE (Semantic mismatch)")
        
        print(f"\n--- Final Results: {passed_tests}/{total_tests} Passed (Top 1) ---")
        s.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_retrieval()
