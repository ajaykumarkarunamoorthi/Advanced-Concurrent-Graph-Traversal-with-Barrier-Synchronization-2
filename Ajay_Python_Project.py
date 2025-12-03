import threading
import queue
import time
from collections import defaultdict, deque
from typing import List, Dict, Set


# ================================================================
# Thread-Safe Graph
# ================================================================
class ConcurrentGraph:
    def __init__(self):
        self.adj = defaultdict(list)
        self.lock = threading.Lock()

    def add_edge(self, u: int, v: int):
        with self.lock:
            self.adj[u].append(v)

    def neighbors(self, u: int):
        return self.adj.get(u, [])


# ================================================================
# Custom Barrier (Reusable)
# ================================================================
class ReusableBarrier:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.count = 0
        self.count_lock = threading.Lock()
        self.phase = threading.Condition(self.count_lock)
        self.current_phase = 0

    def wait(self):
        with self.count_lock:
            local_phase = self.current_phase
            self.count += 1

            # Last thread arrives → reset barrier and advance phase
            if self.count == self.num_threads:
                self.count = 0
                self.current_phase += 1
                self.phase.notify_all()
            else:
                # Other threads wait until phase increments
                while local_phase == self.current_phase:
                    self.phase.wait()


# ================================================================
# Concurrent BFS Worker
# ================================================================
class BFSWorker(threading.Thread):
    def __init__(self, wid, graph, visited, current_level, next_level,
                 work_queue, barrier, lock):
        super().__init__()
        self.wid = wid
        self.graph = graph
        self.visited = visited
        self.current_level = current_level
        self.next_level = next_level
        self.work_queue = work_queue
        self.barrier = barrier
        self.lock = lock
        self.daemon = True

    def run(self):
        while True:
            try:
                node = self.work_queue.get_nowait()
            except queue.Empty:
                break

            for nbr in self.graph.neighbors(node):
                with self.lock:  # protect visited set
                    if nbr not in self.visited:
                        self.visited.add(nbr)
                        self.next_level.append(nbr)

        # Wait for all workers to finish the level
        self.barrier.wait()


# ================================================================
# Concurrent BFS Implementation (Level-Synchronized)
# ================================================================
def concurrent_bfs(graph: ConcurrentGraph, start: int, num_threads=4):
    visited = set([start])
    frontier = [start]

    lock = threading.Lock()  # protects visited set

    while frontier:
        work_queue = queue.Queue()
        for node in frontier:
            work_queue.put(node)

        next_frontier = []

        barrier = ReusableBarrier(num_threads)

        workers = [
            BFSWorker(
                wid=i,
                graph=graph,
                visited=visited,
                current_level=frontier,
                next_level=next_frontier,
                work_queue=work_queue,
                barrier=barrier,
                lock=lock
            )
            for i in range(num_threads)
        ]

        for w in workers:
            w.start()

        for w in workers:
            w.join()

        frontier = next_frontier

    return visited


# ================================================================
# Single-Threaded BFS (Baseline)
# ================================================================
def bfs_single(graph: ConcurrentGraph, start: int):
    visited = set([start])
    q = deque([start])

    while q:
        u = q.popleft()
        for nbr in graph.neighbors(u):
            if nbr not in visited:
                visited.add(nbr)
                q.append(nbr)

    return visited


# ================================================================
# Random Graph Generation
# ================================================================
import random
def generate_graph(num_nodes=10000, edges_per_node=5):
    g = ConcurrentGraph()
    for u in range(num_nodes):
        for _ in range(edges_per_node):
            v = random.randint(0, num_nodes - 1)
            g.add_edge(u, v)
    return g


# ================================================================
# Benchmarking
# ================================================================
def benchmark():
    print("Generating graph...")
    g = generate_graph(num_nodes=15000, edges_per_node=8)

    start_node = 0

    print("\n=== Single-threaded BFS ===")
    t1 = time.time()
    v1 = bfs_single(g, start_node)
    t2 = time.time()
    print(f"Visited nodes: {len(v1)}")
    print(f"Time: {t2 - t1:.4f} sec")

    print("\n=== Concurrent BFS (4 threads) ===")
    t3 = time.time()
    v2 = concurrent_bfs(g, start_node, num_threads=4)
    t4 = time.time()
    print(f"Visited nodes: {len(v2)}")
    print(f"Time: {t4 - t3:.4f} sec")

    print("\nSpeedup:", (t2 - t1) / (t4 - t3))


if __name__ == "__main__":
    benchmark()
