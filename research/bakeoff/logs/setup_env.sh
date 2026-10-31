set -x
cd /root/orbital-dodge/research/bakeoff
/usr/local/bin/uv venv --python 3.12 .venv
. .venv/bin/activate
/usr/local/bin/uv pip install numpy pandas pyarrow scipy scikit-learn lightgbm joblib
/usr/local/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cu124
/usr/local/bin/uv pip install interpret tabpfn huggingface_hub
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
echo ENV_DONE
