@echo off
py -3.9 -m venv venv
call venv\Scripts\activate

echo Instalando dependências principais...
pip install -r requirements.txt

echo Instalando DeepFace sem dependências...
pip install deepface==0.0.79 --no-deps

echo Instalação finalizada. Verificando versão do TensorFlow...
python -c "import tensorflow as tf; print(tf.__version__)"
pause
