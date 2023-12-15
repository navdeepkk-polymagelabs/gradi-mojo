export OMP_NUM_THREADS=32
export PYTHONPATH="/home1/navdeep/work/projects/gradi-mojo:$PYTHONPATH"
dir_name="anim_$(date +"%Y%m%d_%H%M%S")"

mkdir $dir_name
echo "Storing animations in " $dir_name
cd $dir_name


# Generate plots for circle.
mkdir circle
cd circle
python ../../main.py -circle -dim=2 -plots -num-learning-iters=100 -skip-verification -run-cpp
cd ../

# Generate plots for sphere.
mkdir sphere
cd sphere
python ../../main.py -circle -dim=3 -plots -num-learning-iters=300 -skip-verification -num-points-circle=1000 -num-benchmarking-iters=20 -run-cpp
cd ../

# Generate plots for flame_2.
mkdir flame_2
cd flame_2
python ../../main.py -custom-coords-file="../../shapes/flame.csv" -dim=2 -plots -num-learning-iters=200 -skip-verification -run-cpp
cd ../

# Generate plots for flame_3.
mkdir flame_3
cd flame_3
python ../../main.py -custom-coords-file="../../shapes/flame.csv" -dim=3 -plots -num-learning-iters=200 -skip-verification -run-cpp
cd ../

# Generate plots for hipc.
mkdir hipc
cd hipc
python ../../main.py -custom-coords-file="../../shapes/hipc.csv" -num-learning-iters=400 -plots -learning-rate=0.000001 -skip-verification -num-benchmarking-iters=10 -run-cpp
cd ../

cd ../
exit
