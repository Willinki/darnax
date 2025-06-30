import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt


torch.manual_seed(0)


class SubModule(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.depth = depth
        self.layers = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(depth)])

    def forward(self, x):
        return self.layers(x)


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
        self.submodules = nn.ModuleList(
            [SubModule(dim, depth).to(device) for _ in range(num_submodules)]
        )
        self.num_submodules = num_submodules

        # Create CUDA streams once if using GPU
        self.streams = None
        if device == "cuda":
            self.streams = [
                torch.cuda.Stream(device=device) for _ in range(num_submodules)
            ]

        # Create multiprocessing pool once if CPU and using mp
        self.pool = None
        if device == "cpu":
            self.pool = mp.Pool(processes=num_submodules)

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


def benchmark(device, use_parallel, feature_dim, num_submodules, depth, num_iters=10):
    bsize = 16
    x = torch.randn(bsize, feature_dim).to(device)
    model = BenchmarkModel(feature_dim, num_submodules, depth, device)
    model.eval()

    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(num_iters):
        if device == "cuda" and use_parallel:
            model.forward_gpu_streams(x)
        elif device == "cuda" and not use_parallel:
            model.forward_sequential(x)
        elif device == "cpu" and use_parallel:
            model.forward_cpu_mp(x)
        else:
            model.forward_sequential(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    model.close()

    avg_time = (end - start) / num_iters
    return avg_time


def main():
    feature_dims = [512, 1024, 2048, 4096, 6000]
    num_submodules = 10
    depth = 10
    num_iters = 20  # Number of forward passes to average

    results = {
        "GPU (streams)": [],
        "GPU (sequential)": [],
    }

    for dim in feature_dims:
        print(f"\nRunning benchmark for FEATURE_DIM={dim}")

        results["GPU (streams)"].append(
            benchmark(
                "cuda",
                use_parallel=True,
                feature_dim=dim,
                num_submodules=num_submodules,
                depth=depth,
                num_iters=num_iters,
            )
        )
        results["GPU (sequential)"].append(
            benchmark(
                "cuda",
                use_parallel=False,
                feature_dim=dim,
                num_submodules=num_submodules,
                depth=depth,
                num_iters=num_iters,
            )
        )
        # results["CPU (mp)"].append(
        #     benchmark(
        #         "cpu",
        #         use_parallel=True,
        #         feature_dim=dim,
        #         num_submodules=num_submodules,
        #         depth=depth,
        #         num_iters=num_iters,
        #     )
        # )
        # results["CPU (sequential)"].append(
        #     benchmark(
        #         "cpu",
        #         use_parallel=False,
        #         feature_dim=dim,
        #         num_submodules=num_submodules,
        #         depth=depth,
        #         num_iters=num_iters,
        #     )
        # )

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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
