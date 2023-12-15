import argparse
import numpy as np
import jax
import os

from cpp.binding import gradient_descent_cpp, gradient_descent_cpp_to_plot
from python.gradient_descent import gradient_descent, gradient_descent_cache
from python.gradient_descent_native import (
    gradient_descent_native,
    gradient_descent_native_cache,
    PyMatrix,
)
from python.gradient_descent_JAX import (
    gradient_descent_JAX_cpu,
    gradient_descent_JAX_gpu,
    gradient_descent_polyblocks_cpu,
    gradient_descent_polyblocks_single_loop_cpu,
    gradient_descent_polyblocks_single_loop_to_plot_cpu,
    gradient_descent_polyblocks_gpu,
    gradient_descent_polyblocks_single_loop_gpu,
    gradient_descent_polyblocks_single_loop_to_plot_gpu,
    gradient_descent_jax_to_plot_cpu,
    gradient_descent_jax_to_plot_gpu,
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


def benchmark_func(X, D, lr, niter, bench_name, func, benchmark_iters):
    secs = (
        timeit(
            lambda: func(X, D, learning_rate=lr, num_iterations=niter),
            number=benchmark_iters,
        )
        / benchmark_iters
    )
    print(f"Average time {bench_name}: {secs}")
    return secs


def combine_into_two_iframes_and_save(file1, file2, combined_filename='combined.html'):
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{file1} vs {file2}</title>
        <style>
            .container {{
                display: flex;
            }}
            .iframe-container {{
                flex: 1;
                border: 1px solid #ddd;
                margin: 10px;
                height: 100vh;
            }}
            iframe {{
                width: 100%;
                height: 100%;
                border: none; /* Remove iframe border */
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="iframe-container">
                <iframe src="{file1}"></iframe>
            </div>
            <div class="iframe-container">
                <iframe src="{file2}"></iframe>
            </div>
        </div>
    </body>
    </html>
    """
    with open(combined_filename, "w") as file:
        print(
            f"Saving combined animation in {combined_filename}. Save all three generated HTML files to see the animation."
        )
        file.write(combined_html)


def combine_into_three_iframes_and_save(file1, file2, file3, combined_filename='combined.html'):
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{file1} vs {file2} vs {file3}</title>
        <style>
            .container {{
                display: flex;
            }}
            .iframe-container {{
                flex: 1;
                border: 1px solid #ddd;
                margin: 10px;
                height: 100vh;
            }}
            iframe {{
                width: 100%;
                height: 100%;
                border: none; /* Remove iframe border */
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="iframe-container">
                <iframe src="{file1}"></iframe>
            </div>
            <div class="iframe-container">
                <iframe src="{file2}"></iframe>
            </div>
            <div class="iframe-container">
                <iframe src="{file3}"></iframe>
            </div>
        </div>
    </body>
    </html>
    """
    with open(combined_filename, "w") as file:
        print(
            f"Saving combined animation in {combined_filename}. Save all three generated HTML files to see the animation."
        )
        file.write(combined_html)


def benchmarks(
    D,
    dim,
    lr,
    niter,
    benchmark_iters,
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

    # Initial starting point
    np.random.seed(42)
    X = np.random.rand(N, dim)
    D_JAX = jax.device_put(D, device)
    X_JAX = jax.device_put(X, device)

    if verify:
        print("Running verification runs...")

        def run_verification_run(X, D, lr, niter, func):
            return func(X, D, lr, niter)

        all_res = []
        if run_jax:
            func = gradient_descent_JAX_cpu
            if gpu:
                func = gradient_descent_JAX_gpu
            all_res.append(
                (
                    run_verification_run(
                        X_JAX.copy(), D_JAX.copy(), lr, niter, func
                    ),
                    "jax",
                )
            )

        if run_polyblocks:
            if run_polyblocks_double_loop:
                func = gradient_descent_polyblocks_cpu
                if gpu:
                    func = gradient_descent_polyblocks_gpu
                all_res.append(
                    (
                        run_verification_run(
                            X_JAX.copy(), D_JAX.copy(), lr, niter, func
                        ),
                        "polyblocks_double_loop",
                    )
                )
            else:
                func = gradient_descent_polyblocks_single_loop_cpu
                if gpu:
                    func = gradient_descent_polyblocks_single_loop_gpu
                all_res.append(
                    (
                        run_verification_run(
                            X_JAX.copy(), D_JAX.copy(), lr, niter, func
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
                        lr,
                        niter,
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
    speed_up_over_jax = None
    speed_up_over_cpp = None
    if benchmark:
        print("Benchmarking runs...")

        def run_benchmarking_run(X, D, lr, niter, func):
            return run_benchmarking_run

        time_jax = None
        if run_jax:
            # Warm-up.
            func = gradient_descent_JAX_cpu
            if gpu:
                func = gradient_descent_JAX_gpu
            func(
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
                func,
                benchmark_iters,
            )
        time_pb = None
        if run_polyblocks:
            if run_polyblocks_double_loop:
                func = gradient_descent_polyblocks_cpu
                if gpu:
                    func = gradient_descent_polyblocks_gpu
                # Warm-up.
                func(
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
                    func,
                    benchmark_iters,
                )
            else:
                func = gradient_descent_polyblocks_single_loop_cpu
                if gpu:
                    func = gradient_descent_polyblocks_single_loop_gpu
                # Warm-up.
                func(
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
                    func,
                    benchmark_iters,
                )
        time_cpp = None
        if run_cpp:
            time_cpp = benchmark_func(
                X.copy(),
                D.copy(),
                lr,
                niter,
                "CPP",
                gradient_descent_cpp,
                benchmark_iters,
            )

        if time_pb != None and time_jax != None:
            speed_up_over_jax = time_jax / time_pb
            print(f"Speed-up over JAX: {speed_up_over_jax}")

        if time_pb != None and time_cpp != None:
            speed_up_over_cpp = time_cpp / time_pb
            print(f"Speed-up over CPP: {speed_up_over_cpp}")

    ## Visualization
    if plots:
        print("Generating animations...")
        if speed_up_over_jax == None and speed_up_over_cpp == None:
            print(
                "Benchmarking needs to be done before plotting. Enable benchmarking."
            )
            exit(1)
        # Generate Points for PolyBlocks first. We will plot all points with
        # polyblocks as the base reference. Animations will be scaled with
        # polyblocks as the reference.
        func = gradient_descent_polyblocks_single_loop_to_plot_cpu
        if gpu:
            func = gradient_descent_polyblocks_single_loop_to_plot_gpu
        (
            X_res_pb,
            P_res_pb,
            L_res_pb,
        ) = func(
            X_JAX.copy(), D_JAX.copy(), learning_rate=lr, num_iterations=niter
        )
        pb_anim = animate_gradient_descent(
            P_res_pb,
            L_res_pb,
            title="JAX/PolyBlocks",
            filename="polyblocks",
        )

        if run_jax and speed_up_over_jax != None:
            func = gradient_descent_jax_to_plot_cpu
            if gpu:
                func = gradient_descent_jax_to_plot_gpu
            (
                X_res_jax,
                P_res_jax,
                L_res_jax,
            ) = func(
                X_JAX.copy(),
                D_JAX.copy(),
                learning_rate=lr,
                num_iterations=niter,
            )

            jax_anim = animate_gradient_descent(
                P_res_jax,
                L_res_jax,
                title="JAX (JIT)",
                filename="jax",
                duration_scale=speed_up_over_jax,
            )

            combine_into_two_iframes_and_save(jax_anim, pb_anim, 'jax_polyblocks.html')
        elif run_jax and speed_up_over_jax == None:
            print("JAX run is enabled but speed-up is not defined!")
            exit(0)


        if run_cpp and speed_up_over_cpp != None:
            func = gradient_descent_cpp_to_plot
            P_res_cpp = func(
                X.copy(),
                D.copy(),
                learning_rate=lr,
                num_iterations=niter,
            )

            cpp_anim = animate_gradient_descent(
                P_res_cpp,
                L_res_pb,
                title="C++ 16 threads",
                filename="cpp",
                duration_scale=speed_up_over_cpp,
            )

            combine_into_two_iframes_and_save(cpp_anim, pb_anim, 'cpp_polyblocks.html')
        elif run_cpp and speed_up_over_cpp == None:
            print("CPP run is enabled but speed-up is not defined!")
            exit(0)

        if run_cpp and run_jax and speed_up_over_cpp != None and speed_up_over_jax != None:
            combine_into_three_iframes_and_save(jax_anim, cpp_anim, pb_anim, 'cpp_jax_polyblocks.html')


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
        "-custom-coords-file",
        default="",
        help="Custom figure coordinates filename",
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
        default=1000,
        help="Number of learning iterations for gradient descent",
    )
    parser.add_argument(
        "-num-benchmarking-iters",
        type=int,
        default=100,
        help="Number of benchmarking iterations for gradient descent",
    )
    parser.add_argument(
        "-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for gradient descent",
    )

    args = vars(parser.parse_args())

    # Only 2, 3 are valid dims.
    dim = args["dim"]
    assert dim == 2 or dim == 3, "dim must be 2 or 3!"

    circle = args["circle"]
    flame = args["flame"]
    custom_coords_filename = args["custom_coords_file"]

    assert (
        circle or flame or custom_coords_filename != ""
    ), "Either one of circle, flame, or custom-coords-file must be set!"

    if circle:
        n_circle = args["num_points_circle"]
        points = generate_radial_points(n_circle, dim)
    elif flame:
        points = np.loadtxt("./shapes/flame.csv", delimiter=",")
    elif custom_coords_filename != "":
        points = np.loadtxt(custom_coords_filename, delimiter=",")

    # Optimization input
    lr = args["learning_rate"]
    plots = args["plots"]
    niter = args["num_learning_iters"]
    benchmark_iters = args["num_benchmarking_iters"]
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
        benchmark_iters=benchmark_iters,
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
