export OMP_NUM_THREADS=32
export PYTHONPATH="/home1/navdeep/work/projects/gradi-mojo:$PYTHONPATH"

# Set default value for GPU to false
gpu=false

# Check if the "gpu" argument is provided
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            gpu=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

dir_name="anim_$(date +"%Y%m%d_%H%M%S")"

gpu_flag=""
if [ "$gpu" = true ]; then
    echo "GPU is enabled."
    gpu_flag="-gpu"
    dir_name=$dir_name"_gpu"
else
    echo "GPU is not enabled."
fi


mkdir $dir_name
echo "Storing animations in " $dir_name
cd $dir_name


# Generate plots for circle.
mkdir circle
cd circle
python ../../main.py -circle -dim=2 -plots -num-learning-iters=100 -skip-verification -run-cpp $gpu_flag
cd ../

# Generate plots for sphere.
mkdir sphere
cd sphere
python ../../main.py -circle -dim=3 -plots -num-learning-iters=200 -skip-verification -num-points-circle=1000 -learning-rate=0.0001 -num-benchmarking-iters=20 -run-cpp $gpu_flag
cd ../

# Generate plots for flame_2.
mkdir flame_2
cd flame_2
python ../../main.py -custom-coords-file="../../shapes/flame.csv" -dim=2 -plots -num-learning-iters=200 -skip-verification -run-cpp -num-leraning-iters=80 $gpu_flag
cd ../

# Generate plots for flame_3.
mkdir flame_3
cd flame_3
python ../../main.py -custom-coords-file="../../shapes/flame.csv" -dim=3 -plots -num-learning-iters=200 -skip-verification -run-cpp $gpu_flag
cd ../

# Generate plots for hipc.
mkdir hipc
cd hipc
python ../../main.py -custom-coords-file="../../shapes/hipc.csv" -num-learning-iters=400 -plots -learning-rate=0.000001 -skip-verification -num-benchmarking-iters=10 -run-cpp $gpu_flag
cd ../

cd ../
exit
