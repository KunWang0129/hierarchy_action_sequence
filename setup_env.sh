module purge
module load anaconda3/2025.6
conda create --name action_seq python=3.11
conda activate action_seq

pip install --upgrade pip
pip install -r requirements.txt