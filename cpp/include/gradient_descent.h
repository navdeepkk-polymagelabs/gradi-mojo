// Input type definitions
struct InputBindingInterface {
    int N;
    int dim;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> D;
    float learning_rate;
    int num_iterations;
};

// Output type definitions
struct OutputBindingInterface {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
};

// Output type definitions
struct PlotOutputBindingInterface {
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> all_X;
};


// Bounded function
OutputBindingInterface gradient_descent(InputBindingInterface input);

// Bounded function
PlotOutputBindingInterface gradient_descent_to_plot_cpp(InputBindingInterface input);
