import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt
from typing import List

torch.manual_seed(0)


class SubModule(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(depth)])

    def forward(self, x):
        return self.layers(x)


def make_scripted_submodule(dim, depth):
    class ScriptedSubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(depth)])

        def forward(self, x):
            return self.layers(x)

    return torch.jit.script(ScriptedSubModule())


@torch.jit.script
def forward_jit_fork(
    submodules: List[torch.jit.ScriptModule], x: torch.Tensor
) -> torch.Tensor:
    futures = [torch.jit.fork(mod, x) for mod in submodules]
    outputs = [torch.jit.wait(f) for f in futures]
    return torch.stack(outputs, dim=0)


def _cpu_worker(args):
    state_dict, x, dim, depth = args
    m = SubModule(dim, depth)
    m.load_state_dict(state_dict)
    m.eval()
    with torch.no_grad():
        return m(x)


class BenchmarkModel(nn.Module):
    def __init__(self, dim, num_submodules, depth, device):
        super().__init__()
        self.device = device
        self.dim = dim
        self.depth = depth
        self.num_submodules = num_submodules

        self.submodules = nn.ModuleList(
            [SubModule(dim, depth).to(device) for _ in range(num_submodules)]
        )

        # CUDA streams for parallel GPU
        self.streams = None
        if device == "cuda":
            self.streams = [
                torch.cuda.Stream(device=device) for _ in range(num_submodules)
            ]

        # CPU multiprocessing pool
        self.pool = None
        if device == "cpu":
            self.pool = mp.Pool(processes=num_submodules)

        # JIT ScriptModules for fork
        self.jit_submodules: List[torch.jit.ScriptModule] = []
        if device == "cuda":
            for _ in range(num_submodules):
                mod = make_scripted_submodule(dim, depth).to(device)
                self.jit_submodules.append(mod)

    def forward_sequential(self, x):
        return torch.stack([m(x) for m in self.submodules], dim=0)

    def forward_gpu_streams(self, x):
        outputs = [None] * self.num_submodules
        for i, (mod, stream) in enumerate(zip(self.submodules, self.streams)):
            with torch.cuda.stream(stream):
                outputs[i] = mod(x)
        torch.cuda.synchronize()
        return torch.stack(outputs, dim=0)

    def forward_cpu_mp(self, x):
        args = [
            (m.cpu().state_dict(), x.cpu(), self.dim, self.depth)
            for m in self.submodules
        ]
        results = self.pool.map(_cpu_worker, args)
        return torch.stack(results, dim=0)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def benchmark(
    device,
    use_parallel,
    feature_dim,
    num_submodules,
    depth,
    num_iters=10,
    mode="default",
):
    bsize = 16
    x = torch.randn(bsize, feature_dim).to(device)
    model = BenchmarkModel(feature_dim, num_submodules, depth, device)
    model.eval()

    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start = time.time()
        if mode == "jit_fork":
            forward_jit_fork(model.jit_submodules, x)
        elif device == "cuda" and use_parallel:
            model.forward_gpu_streams(x)
        elif device == "cuda" and not use_parallel:
            model.forward_sequential(x)
        elif device == "cpu" and use_parallel:
            model.forward_cpu_mp(x)
        else:
            model.forward_sequential(x)
        end = time.time()
    times.append(end - start)

    if device == "cuda":
        torch.cuda.synchronize()

    model.close()

    times.sort()
    avg_time = times[len(times) // 2]
    return avg_time


def main():
    feature_dims = [256, 512, 1024, 2048]
    num_submodules = 10
    depth = 10
    num_iters = 100  # Number of forward passes to average

    results = {
        "GPU (sequential)": [],
        "GPU (streams)": [],
        "GPU (jit_fork)": [],
    }

    for dim in feature_dims:
        print(f"\nRunning benchmark for FEATURE_DIM={dim}")

        results["GPU (sequential)"].append(
            benchmark(
                "cuda",
                use_parallel=False,
                feature_dim=dim,
                num_submodules=num_submodules,
                depth=depth,
                num_iters=num_iters,
                mode="default",
            )
        )

        results["GPU (streams)"].append(
            benchmark(
                "cuda",
                use_parallel=True,
                feature_dim=dim,
                num_submodules=num_submodules,
                depth=depth,
                num_iters=num_iters,
                mode="default",
            )
        )

        results["GPU (jit_fork)"].append(
            benchmark(
                "cuda",
                use_parallel=True,
                feature_dim=dim,
                num_submodules=num_submodules,
                depth=depth,
                num_iters=num_iters,
                mode="jit_fork",
            )
        )

    # Plotting
    plt.figure(figsize=(10, 6))
    for label, times in results.items():
        plt.plot(feature_dims, times, label=label, marker="o")

    plt.xlabel("FEATURE_DIM")
    plt.ylabel("Average Forward Pass Time (s)")
    plt.title(f"Benchmark vs FEATURE_DIM (averaged over {num_iters} runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_featuredim.png")
    print("\nBenchmark plot saved as 'benchmark_featuredim.png'")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
