# Advanced-Concurrent-Graph-Traversal-with-Barrier-Synchronization-2
Concurrent Graph Traversal (Python)

This project implements a multi-threaded Breadth-First Search (BFS) in Python using a custom barrier synchronization mechanism.
It also includes a single-threaded BFS and a performance comparison.

Features

Thread-safe graph data structure

Custom reusable barrier for synchronizing BFS levels

Multi-threaded BFS using worker threads

Single-threaded BFS for comparison

Random graph generator

Benchmark showing runtime of both versions
Output
Generating graph...

=== Single-threaded BFS ===
Visited nodes: 14992
Time: 0.0070 sec

=== Concurrent BFS (4 threads) ===
Visited nodes: 14992
Time: 0.0499 sec

Speedup: 0.14
