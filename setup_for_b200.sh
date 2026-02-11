conda init
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n soft_grpo python=3.11.13 -y
conda init && eval "$(conda shell.bash hook)" && conda activate soft_grpo
pip install pip==25.2
pip install tensorboard==2.20.0 accelerate==1.10.1 torch_memory_saver==0.0.8 uvloop==0.21.0 jsonlines math_verify openai

cd Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ../..

cd verl-0.4.x
pip3 install -e .
cd ..


pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install transformers==4.54.0
pip install sgl-kernel==0.3.16.post6

pip install flash_attn==2.7.3 --no-build-isolation
pip install --upgrade flashinfer-python
pip install triton==3.3.1

