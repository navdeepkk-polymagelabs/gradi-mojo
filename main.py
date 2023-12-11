import numpy as np

from cpp.binding import gradient_descent_cpp
from python.gradient_descent import gradient_descent, gradient_descent_cache
from python.gradient_descent_native import (
    gradient_descent_native,
    gradient_descent_native_cache,
    PyMatrix,
)
from python.gradient_descent_JAX import (
    gradient_descent_JAX,
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
import jax

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


def benchmark_gradient_descent_native(X_native, D_native, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent_native(
                X_native, D_native, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time python native: {secs}")
    return secs


def benchmark_gradient_descent(X, D, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent(
                X, D, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time python numpy: {secs}")
    return secs


def benchmark_gradient_descent_JAX(X, D, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent_JAX(
                X, D, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time JAX: {secs}")
    return secs


def benchmark_gradient_descent_polyblocks(X, D, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent_polyblocks(
                X, D, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time polyblocks: {secs}")
    return secs


def benchmark_gradient_descent_polyblocks_single_loop(X, D, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent_polyblocks_single_loop(
                X, D, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time polyblocks with single loop: {secs}")
    return secs


def benchmark_gradient_descent_cpp(X, D, lr, niter):
    secs = (
        timeit(
            lambda: gradient_descent_cpp(
                X, D, learning_rate=lr, num_iterations=niter
            ),
            number=NUM_ITERS,
        )
        / NUM_ITERS
    )
    print(f"Average time C++ binding: {secs}")
    return secs


def benchmarks(D, dim, lr, niter, plots=True):
    N = len(D)
    D = np.array(D, dtype=np.float64)
    D_native = PyMatrix(D.tolist(), N, N)

    # Initial starting point
    np.random.seed(42)
    X = np.random.rand(N, dim)
    X_native = PyMatrix(X.tolist(), N, dim)
    D_JAX = jax.device_put(D, jax.devices("cpu")[0])
    X_JAX = jax.device_put(X, jax.devices("cpu")[0])

    ### Without visuals
    # p1 = gradient_descent_native(X_native.copy(), D_native, learning_rate=lr, num_iterations=niter)
    # p2 = gradient_descent(X.copy(), D, learning_rate=lr, num_iterations=niter)
    p3 = gradient_descent_JAX(
        X_JAX.copy(), D_JAX, learning_rate=lr, num_iterations=niter
    )
    # p4 = gradient_descent_polyblocks(X_JAX.copy(), D_JAX, learning_rate=lr, num_iterations=niter)
    p5 = gradient_descent_polyblocks_single_loop(
        X_JAX.copy(), D_JAX, learning_rate=lr, num_iterations=niter
    )
    # p_cpp = gradient_descent_cpp(X.copy(), D, learning_rate=lr, num_iterations=niter)

    if not np.allclose(p3, p5, atol=0.0005):
        print("Verification failed")
        exit(1)
    print("Verification successful")

    ### Benchmarks
    # benchmark_gradient_descent_native(X_native.copy(), D_native, lr=lr, niter=niter)
    # benchmark_gradient_descent(X.copy(), D, lr=lr, niter=niter)
    time_jax = benchmark_gradient_descent_JAX(
        X_JAX.copy(), D_JAX, lr=lr, niter=niter
    )
    # benchmark_gradient_descent_polyblocks(X_JAX.copy(), D_JAX, lr=lr, niter=niter)
    time_pb = benchmark_gradient_descent_polyblocks_single_loop(
        X_JAX.copy(), D_JAX, lr=lr, niter=niter
    )
    # benchmark_gradient_descent_cpp(X.copy(), D, lr=lr, niter=niter)

    speed_up = time_jax / time_pb

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
    # Create optimization target
    n_circle = 100
    dim_circle = 2
    points = generate_radial_points(n_circle, dim_circle)  # circle/spher3
    # points = np.loadtxt("./shapes/modular.csv", delimiter=",")      # modular (N = 1000)
    # points = np.loadtxt("./shapes/flame.csv", delimiter=",")        # flame (N = 307)

    # Optimization input
    dim = 2
    lr = 0.001
    niter = 200
    plots = True

    benchmarks(
        D=generate_distance_matrix(points),
        dim=dim,
        lr=lr,
        niter=niter,
        plots=plots,
    )
