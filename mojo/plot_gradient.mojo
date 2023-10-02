from python.python import Python
from runtime.llcl import Runtime

from mojo.gradient_descent import Matrix, gradient_descent, compute_gradient, loss, type



fn plot_gradient_descent_cache(inout X: Matrix, D: Matrix, learning_rate: type = 0.0001, num_iterations: Int = 1000):
    let N = X.rows
    let dim = X.cols
    var grad = Matrix(N, dim)

    var positions_over_time: PythonObject = []
    var loss_over_time: PythonObject = []

    with Runtime() as rt:
        try:
            for i in range(num_iterations):
                
                positions_over_time += flatten(X)                         # ? Can't seem to find another way but to flatten and reshape
                loss_over_time += [loss(X, D).cast[DType.float64]()]      # ? Mandatory cast to float64
                
                compute_gradient(grad, X, D, rt)
                for r in range(X.rows):
                    for c in range(X.cols):
                        X[r, c] -= learning_rate * grad[r, c]

            positions_over_time += flatten(X)
            loss_over_time += [loss(X, D).cast[DType.float64]()]
        except e:
            print(e.value)

    # Flat array shape
    let shape: Tuple[Int, Int, Int] = (num_iterations + 1, X.rows, X.cols)
    
    try:
        Python.add_to_path("./python")
        let np = Python.import_module("numpy")
        let utils = Python.import_module("utils")
    
        positions_over_time = np.reshape(positions_over_time, shape)
        let a = utils.plot_gradient_descent(positions_over_time, loss_over_time)
    except e:
        print(e.value)


fn flatten(X: Matrix) -> PythonObject:
    var flat_array: PythonObject = []
    for i in range(X.rows):
        for j in range(X.cols):
            try:
                flat_array += [X[i, j].cast[DType.float64]()]
            except e:
                print(e.value)

    return flat_array