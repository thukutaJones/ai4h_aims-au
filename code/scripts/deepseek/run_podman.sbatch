#!/bin/bash
#SBATCH --job-name=podman_job
#SBATCH --output=podman_job_%j.out
#SBATCH --error=podman_job_%j.err
#SBATCH --time=44:00:00  # Max time per job
#SBATCH --ntasks-per-node=1
#SBATCH -n 1
#SBATCH --mem=1000G
#SBATCH --gpus-per-task=h100:4
#SBATCH --cpus-per-task=50
#SBATCH --reservation=AMLRT
#SBATCH --partition=staff-amlrt
#SBATCH --constraint=nvlink

# Load necessary modules (if needed)
conda init
conda activate deepseek

podman ps -q | xargs -r podman kill

PODMAN_CMD="podman run -v /network/projects/amlrt/qut01-aims/HF_HOME/DeepSeek-R1-GGUF:/models \
--rm --device nvidia.com/gpu=all \
-p 8080:8080 \
ghcr.io/ggerganov/llama.cpp:full-cuda \
--server -m /models/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
--ctx-size 8192 --temp 0.6 --threads 16 --n-gpu-layers 62 --cache-type-k q4_0 \
--prio 2 --host 0.0.0.0 --seed 3407"

# If setup through llama.cpp, use the following command
# PODMAN_CMD="./llama.cpp/llama-server \
#    --model /network/projects/amlrt/qut01-aims/HF_HOME/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
#    --cache-type-k q4_0 \
#    --threads 16 \
#    --n-gpu-layers 62 \
#    --ctx-size 8192 \
#    --temp 0.6 \
#    --prio 2 \
#    --seed 3407 \
#    --host 0.0.0.0 --port 8090"

# Run the Podman container
echo "Starting Podman container..."
eval $PODMAN_CMD &  # Run in the background
PODMAN_PID=$!

# Wait for 20 minutes (1200 seconds) to allow the model to load
echo "Waiting 20 minutes for model to load..."
sleep 600

conda deactivate
conda init
conda activate qut02

cd /home/mila/a/akshatha.arodi-nagaraja/qut01-aims/scripts/deepseek

# Run Python scripts after waiting
# Run Python scripts in parallel (from criteria 0 to 9)
echo "Running Python scripts in parallel..."
for i in 0 1 3 4 5 6 7 8 9; do
	echo "Starting run_deepseek_r1.py with argument $i"
	python run_deepseek_r1.py "$i" "../data/contextualized_data/AU_context0.csv" &
done


# Wait for all Python scripts to complete
wait
echo "All Python scripts finished."


# Kill the Podman process
echo "Killing Podman container..."
kill $PODMAN_PID
sleep 5  # Give it a moment to shut down

# Force kill if it's still running
if ps -p $PODMAN_PID > /dev/null; then
    echo "Force killing Podman..."
    kill -9 $PODMAN_PID
fi


echo "Job complete. Exiting."
