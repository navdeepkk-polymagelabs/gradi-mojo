import argparse
import numpy as np
import jax
import os

from cpp.binding import gradient_descent_cpp
from python.gradient_descent import gradient_descent, gradient_descent_cache
from python.gradient_descent_native import (
    gradient_descent_native,
    gradient_descent_native_cache,
    PyMatrix,
)
from python.gradient_descent_JAX import (
    gradient_descent_JAX_cpu,
    gradient_descent_JAX_gpu,
    gradient_descent_polyblocks,
    gradient_descent_polyblocks_single_loop,
    gradient_descent_polyblocks_single_loop_to_plot,
    gradient_descent_jax_single_loop_to_plot,
    gradient_descent_cache_JAX,
)
from python.visuals import (
    plot_gradient_descent,
    plot_gradient_descent_2D,
    animate_gradient_descent,
)

from timeit import timeit


def generate_radial_points(N, dim):
    r = 0.5
    points = []
    if dim == 2:
        for i in range(N):
            angle = 2 * np.pi * i / N
            points.append([r * np.cos(angle), r * np.sin(angle)])
    elif dim == 3:
        for i in range(N):
            phi = np.arccos(1 - 2 * (i / N))
            theta = np.sqrt(N * np.pi) * phi
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append([x, y, z])
    else:
        raise ValueError("Only supports 2D and 3D")

    return points


def generate_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


NUM_ITERS = 100


def benchmark_func(X, D, lr, niter, bench_name, func):
    secs = (
        timeit(
            lambda: func(X, D, learning_rate=lr, num_iterations=niter),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time {bench_name}: {secs}")
    return secs


def benchmarks(
    D,
    dim,
    lr,
    niter,
    gpu,
    verify,
    benchmark,
    atol,
    run_jax,
    run_polyblocks,
    run_polyblocks_double_loop,
    run_cpp,
    plots,
):
    if gpu:
        device = jax.devices("gpu")[0]
        print("Running on the GPU")
        os.environ["JAX_PLATFORMS"] = "gpu"
    else:
        device = jax.devices("cpu")[0]
        print("Running on the CPU")
        os.environ["JAX_PLATFORMS"] = "cpu"

    N = len(D)
    D = np.array(D, dtype=np.float64)
    D_native = PyMatrix(D.tolist(), N, N)

    # Initial starting point
    np.random.seed(42)
    X = np.random.rand(N, dim)
    X_native = PyMatrix(X.tolist(), N, dim)
    D_JAX = jax.device_put(D, device)
    X_JAX = jax.device_put(X, device)

    if verify:
        print("Running verification runs...")
        all_res = []
        if run_jax:
            if not gpu:
                all_res.append(
                    (
                        gradient_descent_JAX_cpu(
                            X_JAX.copy(),
                            D_JAX.copy(),
                            learning_rate=lr,
                            num_iterations=niter,
                        ),
                        "jax",
                    )
                )
            else:
                all_res.append(
                    (
                        gradient_descent_JAX_gpu(
                            X_JAX.copy(),
                            D_JAX.copy(),
                            learning_rate=lr,
                            num_iterations=niter,
                        ),
                        "jax",
                    )
                )

        if run_polyblocks:
            if run_polyblocks_double_loop:
                all_res.append(
                    (
                        gradient_descent_polyblocks(
                            X_JAX.copy(),
                            D_JAX.copy(),
                            learning_rate=lr,
                            num_iterations=niter,
                        ),
                        "polyblocks_double_loop",
                    )
                )
            else:
                all_res.append(
                    (
                        gradient_descent_polyblocks_single_loop(
                            X_JAX.copy(),
                            D_JAX.copy(),
                            learning_rate=lr,
                            num_iterations=niter,
                        ),
                        "polyblocks_single_loop",
                    )
                )
        if run_cpp:
            all_res.append(
                (
                    gradient_descent_cpp(
                        X.copy(),
                        D.copy(),
                        learning_rate=lr,
                        num_iterations=niter,
                    ),
                    "cpp",
                )
            )

        def verify(a, b):
            return np.allclose(a, b, atol=atol)

        # Verify all outputs.
        for a in all_res:
            for b in all_res:
                if not verify(a[0], b[0]):
                    print(f"Verification failed for {a[1]} and {b[1]}")
                    exit(1)
        print("Verification successful")

    ### Benchmarks
    if benchmark:
        print("Benchmarking runs...")
        if run_jax:
            # Warm-up.
            if not gpu:
                gradient_descent_JAX_cpu(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    learning_rate=lr,
                    num_iterations=niter,
                )
                time_jax = benchmark_func(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    lr,
                    niter,
                    "JAX",
                    gradient_descent_JAX_cpu,
                )
            else:
                gradient_descent_JAX_gpu(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    learning_rate=lr,
                    num_iterations=niter,
                )
                time_jax = benchmark_func(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    lr,
                    niter,
                    "JAX",
                    gradient_descent_JAX_gpu,
                )
        if run_polyblocks:
            if run_polyblocks_double_loop:
                # Warm-up.
                gradient_descent_polyblocks(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    learning_rate=lr,
                    num_iterations=niter,
                )
                time_pb = benchmark_func(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    lr,
                    niter,
                    "Polyblocks",
                    gradient_descent_polyblocks,
                )
            else:
                # Warm-up.
                gradient_descent_polyblocks_single_loop(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    learning_rate=lr,
                    num_iterations=niter,
                )
                time_pb = benchmark_func(
                    X_JAX.copy(),
                    D_JAX.copy(),
                    lr,
                    niter,
                    "Polyblocks",
                    gradient_descent_polyblocks_single_loop,
                )
        if run_cpp:
            time_cpp = benchmark_func(
                X_JAX.copy(),
                D_JAX.copy(),
                lr,
                niter,
                "CPP",
                gradient_descent_cpp,
            )

        speed_up_over_jax = time_jax / time_pb
        print(f'Speed-up over JAX: {speed_up_over_jax}')

    ## Visualization
    if plots:
        # P, L = gradient_descent_cache(X.copy(), D, learning_rate=lr, num_iterations=niter)
        # plot_gradient_descent_2D(P, L, title="Gradient Descent in python numpy")
        # plot_gradient_descent(P, L, title="Gradient Descent in python numpy")

        # P_native, L_native = gradient_descent_native_cache(X_native.copy(), D_native, learning_rate=lr, num_iterations=niter)
        # plot_gradient_descent(P_native, L_native, title="Gradient Descent in native python")

        X_res, P_res, L_res = gradient_descent_polyblocks_single_loop_to_plot(
            X_JAX.copy(), D_JAX.copy(), learning_rate=lr, num_iterations=niter
        )
        (
            X_res_jax,
            P_res_jax,
            L_res_jax,
        ) = gradient_descent_jax_single_loop_to_plot(
            X_JAX.copy(), D_JAX.copy(), learning_rate=lr, num_iterations=niter
        )

        # To make the animation informative we adjust the frames as per the speed_up.
        if speed_up > 1.0:
            import math

            # Adjust points for the JAX run.
            P_res_jax_to_use = []
            L_res_jax_to_use = []
            for p, l in zip(P_res_jax, L_res_jax):
                for i in range(math.ceil(speed_up)):
                    P_res_jax_to_use.append(p)
                    L_res_jax_to_use.append(l)
            P_res_jax = P_res_jax_to_use
            L_res_jax = L_res_jax_to_use

        if speed_up <= 1.0:
            import math

            # Adjust points for the JAX run.
            P_res_to_use = []
            L_res_to_use = []
            for p, l in zip(P_res, L_res):
                for i in range(math.ceil(speed_up)):
                    P_res_to_use.append(p)
                    L_res_to_use.append(l)
            P_res = P_res_to_use
            L_res = L_res_to_use

        animate_gradient_descent(P_res, L_res, title="circle 2 polyblocks")
        animate_gradient_descent(P_res_jax, L_res_jax, title="circle 2 jax")

        # (cache function not implemented: Can only plot final value)
        # plot_gradient_descent(p_cpp, -1, title="Gradient Descent in C++")

        # animate_gradient_descent(P, L, trace=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gradient descent via different execution backends."
    )
    parser.add_argument(
        "-gpu",
        action="store_true",
        default=False,
        help="Enable GPU execution",
    )
    parser.add_argument(
        "-plots",
        action="store_true",
        default=False,
        help="Enable plot generation",
    )
    parser.add_argument(
        "-circle",
        action="store_true",
        default=False,
        help="Run gradient descent for circle",
    )
    parser.add_argument(
        "-skip-verification",
        action="store_true",
        default=False,
        help="Do not verify results before benchmarking",
    )
    parser.add_argument(
        "-skip-benchmark",
        action="store_true",
        default=False,
        help="Do not benchmark",
    )
    parser.add_argument(
        "-atol", type=float, default=1e-04 * 5, help="atol for verification"
    )
    parser.add_argument(
        "-flame",
        action="store_true",
        default=False,
        help="Run gradient descent for flame",
    )
    parser.add_argument(
        "-skip-jax",
        action="store_true",
        default=False,
        help="Skip jax execution",
    )
    parser.add_argument(
        "-run-cpp",
        action="store_true",
        default=False,
        help="Enable CPP execution",
    )
    parser.add_argument(
        "-skip-polyblocks",
        action="store_true",
        default=False,
        help="Skip polyblocks execution",
    )
    parser.add_argument(
        "-run-polyblocks-double-loop",
        action="store_true",
        default=False,
        help="Run polyblocks double loop version. By default we run the single loop version as that is faster.",
    )
    parser.add_argument(
        "-dim", type=int, default=2, help="Dimensionality of input points"
    )
    parser.add_argument(
        "-num-points-circle",
        type=int,
        default=100,
        help="Number of points in the circle",
    )
    parser.add_argument(
        "-num-learning-iters",
        type=int,
        default=100,
        help="Number of learning iterations for gradient descent",
    )

    args = vars(parser.parse_args())

    # Only 2, 3 are valid dims.
    dim = args["dim"]
    assert dim == 2 or dim == 3, "dim must be 2 or 3!"

    circle = args["circle"]
    flame = args["flame"]

    assert circle or flame, "Either one of circle or flame must be set!"

    if circle:
        n_circle = args["num_points_circle"]
        points = generate_radial_points(n_circle, dim)
    elif flame:
        points = np.loadtxt("./shapes/flame.csv", delimiter=",")

    # Optimization input
    lr = 0.001
    niter = 200
    plots = args["plots"]
    niter = args["num_learning_iters"]
    gpu = args["gpu"]
    verify = not args["skip_verification"]
    atol = args["atol"]
    run_jax = not args["skip_jax"]
    run_polyblocks = not args["skip_polyblocks"]
    run_polyblocks_double_loop = args["run_polyblocks_double_loop"]
    run_cpp = args["run_cpp"]
    benchmark = not args["skip_benchmark"]

    print(f"Running for {niter} iterations with learning rate {lr}")

    benchmarks(
        D=generate_distance_matrix(points),
        dim=dim,
        lr=lr,
        niter=niter,
        gpu=gpu,
        verify=verify,
        benchmark=benchmark,
        atol=atol,
        run_jax=run_jax,
        run_polyblocks=run_polyblocks,
        run_polyblocks_double_loop=run_polyblocks_double_loop,
        run_cpp=run_cpp,
        plots=plots,
    )
