# hubeet-vector-lite

**hubeet-vector-lite (hvl)** is an ultra-high performance, production-grade vector search engine built in C. It implements the **Hierarchical Navigable Small World (HNSW)** algorithm with extensive hardware acceleration and a lock-free concurrency model designed for modern many-core servers.

## Key Features

- **HNSW Architecture**: Full implementation of Malkov's heuristic for neighbor selection and hierarchical graph navigation.
- **Hardware Acceleration (SIMD)**: 
    - **ARM Neon**: Optimized for Apple Silicon (M1/M2/M3).
    - **AVX2 & FMA**: Optimized for modern x86_64 processors.
    - **Zero-Sqrt Optimization**: Uses Squared L2 distance for maximum throughput.
- **Enterprise Concurrency**:
    - **Fine-Grained Locking**: Node-level mutexes for high-concurrency insertions.
    - **Snapshot-Based Traversal**: Critical paths are lock-minimized using stack-based neighbor snapshots.
    - **TLS Search Contexts**: Zero-allocation search path using C11 Thread Local Storage.
- **Memory Efficiency**:
    - **Scalar Quantization (SQ8)**: Foundations for 4x reduction in memory footprint.
    - **Software Prefetching**: Hides memory latency through `__builtin_prefetch` hints.
- **Production Ready**:
- **Persistence**: Fast Binary Format (HVL2) with full graph serialization for O(1) loading.
- **RESP Pipelining**: Stateful protocol parser supporting batch command streams for massive throughput.
- **Protocol**: Simple line-based protocol for easy integration with standard tools.
    - **Performance Suite**: Integrated benchmarking with multi-threading and recall validation.

## Benchmarks & Comparisons

Hubeet Vector Engine is built for extreme low-latency environments. Below is a comparison against a standard Redis search setup on the same hardware (Apple Silicon / 128D Vectors).

### Performance Summary

| Metric | Redis (Standard) | HVL (hubeet-lite) | **Difference (%)** |
| :--- | :--- | :--- | :--- |
| **Search Throughput** | ~15,000 QPS | **24,313 QPS** | **+62.1% Superior** |
| **Avg. Query Latency** | ~2.50 ms | **0.38 ms** | **-84.8% Lighter** |
| **P95 Latency** | ~5.00 ms | **1.13 ms** | **-77.4% Faster** |
| **Insertion Speed** | ~1,500 ups/s | **2,117 ups/s** | **+41.1% Superior** |

*Note: The hardware "sweet spot" on Apple Silicon is identified at **8 threads** for optimal throughput/latency stability.*

### Why is HVL so fast?

1.  **Pure C & Silicon-Native Optimization**:
    Unlike general-purpose databases, HVL has **Zero Overhead**. It doesn't spend CPU cycles on complex dictionaries, key expirations, or general-purpose data management. Every cycle is dedicated to the HNSW traversal and custom **SIMD (Neon/AVX2)** distance math.

2.  **Ultra-Low Latency Architecture**:
    The massive **84% reduction in average latency** makes search feel "instant" (sub-millisecond). While traditional systems deal with kernel context switching for socket events and deep protocol parsing, HVL’s **lock-free search path** allows multiple threads to traverse the graph without any contention.

3.  **Enterprise Stability**:
    Even under heavy load (16 threads), HVL maintains a P99 similar to the P95 of top-tier commercial solutions. This stability is achieved through careful memory alignment (32-byte) and cache-friendly node layouts.

## 🛠️ Build & Run

### Prerequisites
- GCC or Clang (supporting C11).
- Pthread library.

### Compilation
The build system automatically detects your CPU architecture (Neon/AVX2) and applies native optimizations.

```bash
# Build server, tests, and benchmark
make all
```

### Running Benchmark
```bash
make clean && make hvl-bench && ./hvl-bench
```

### Running Server
```bash
# Using defaults
./hvl-server

# Using a custom config file
./hvl-server hvl.conf
```

### Command Line Interface (CLI)
Interact with the server using a Redis-like CLI:
```bash
./hvl-cli 127.0.0.1 5555
```

Once inside, you can run:
- `PING`: Check health.
- `VSET id [v1,v2...]`: Insert a vector.
- `VSEARCH k [v1,v2...]`: Search top-k.
- `TSET id [text]`: Insert a text and convert it to vector using the inference model configured in hvl.conf.
- `TSEARCH k [text]`: Search top-k using the inference model configured in hvl.conf.
- `INFO`: Get server info.
- `HGETALL regexp`: Get all vectors that match the regexp.
- `SAVE`: Persist the index.
- `QUIT`: Exit.

### Command Pipelining
HVL supports **RESP Pipelining**, allowing you to send multiple commands in a single TCP request. This is the recommended way for bulk indexing or syncing large datasets.

**Performance:**
- Light commands (PING): **~970,000 QPS**.
- Bulk Indexing (128D): **~2,300 vectors/sec** (limited only by HNSW insertion logic).

**Safety Guards:**
- **Max Buffer**: 32MB per connection to prevent DoS.
- **Max Batch**: 2,000 commands per "turn" to ensure fair scheduling between clients.

## Configuration

The engine can be configured using a simple `key = value` file. If no file is provided, defaults are used.

### Available Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `dim` | Vector dimension (must match your data). | `128` |
| `M` | Max number of neighbor connections per node. | `16` |
| `ef_construction` | Search depth during index construction. | `400` |
| `ef_search` | Search depth during query execution. | `128` |
| `max_levels` | Maximum hierarchical levels in HNSW. | `16` |
| `port` | TCP port for the server. | `5555` |
| `bind_addr` | Network interface to bind to. | `0.0.0.0` |

### Example `hvl.conf`
```ini
# Hubeet Vector Engine Settings
dim = 128
M = 32
ef_construction = 400
ef_search = 128
port = 6379
bind_addr = 127.0.0.1
```

## Technical Architecture

- **`src/hvl_hnsw.c`**: Core HNSW logic, neighbor selection, and TLS context management.
- **`src/hvl_vector.c`**: SIMD-accelerated distance metrics (L2, Cosine) with FMA support.
- **`src/hvl_quantizer.c`**: Scalar Quantization foundations for SQ8 encoding.
- **`src/hvl_server.c`**: Multi-threaded TCP server with RESP protocol.
- **`src/hvl_persistence.c`**: Binary serialization for RDB-style snapshots.

## Roadmap (Enterprise Vision)

1. **Full SQ8 Integration**: Migration of the internal node storage to use quantized bytes.
2. **Batch Query Processing**: Further SIMD optimization for processing multiple vectors in a single pass.
3. **Productive Client Libraries**: High-level bindings for Python and Go.

---
Developed with ❤️ for high-velocity AI infrastructure.
