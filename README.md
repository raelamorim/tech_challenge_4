# 🎥 Reconhecimento Facial com MediaPipe e OpenCV

Este projeto realiza a detecção de múltiplos rostos em tempo real utilizando a biblioteca **MediaPipe FaceMesh** e **OpenCV**, com suporte tanto para **webcam** quanto para **arquivos de vídeo**.

## 📁 Estrutura de Pastas
facial_detection/</br>
├── 📄 main.py</br>
├── 📁 app/</br>
│ ├── 📄 detect_activity.py</br>
│ ├── 📄 face_mesh_service.py</br>
│ ├── 📄 pose_service.py</br>
│ └── 📄 video_processor.py</br>
├── 📁 assets/</br>
│ ├── 🎞️ example.mp4</br>
│ └── 🎞️ output_video_activity.mp4</br>
├── 📁 infra/</br>
│ └── 📄 video_capture.py</br>
├── 📄 requirements.txt</br>
├── ⚙️ install.bat</br>
└── 📄 README.md

## 📂 Explicação da Estrutura de Pastas

#### 📁 `facial_detection/`
Diretório raiz do projeto. Contém os arquivos principais da aplicação e a organização modular em subpastas.

#### 📄 `main.py`
Arquivo principal que executa a aplicação. Faz uso dos módulos de `app`, `infra` e `assets`.

#### 📁 `app/`
Contém a lógica central de processamento e detecção facial e corporal:

- 📄 `detect_activity.py`: Detecta atividades como levantar o braço, dançar e movimento das mãos com base nas poses do corpo.
- 📄 `face_mesh_service.py`: Responsável por configurar e executar o modelo FaceMesh da MediaPipe.
- 📄 `pose_service.py`: Manipula a detecção de poses corporais usando o MediaPipe Pose.
- 📄 `video_processor.py`: Orquestra o processamento de vídeo quadro a quadro, chamando os módulos apropriados.

#### 📁 `assets/`
Contém arquivos de mídia utilizados para teste ou demonstração:

- 🎞️ `example.mp4`: Vídeo de entrada usado para testar as detecções.
- 🎞️ `output_video_activity.mp4`: Vídeo gerado com as anotações de detecção.

#### 📁 `infra/`
Contém abstrações e serviços de infraestrutura:

- 📄 `video_capture.py`: Gerencia a captura de vídeo (ex: webcam ou arquivo de vídeo).

#### 📄 `requirements.txt`
Lista as dependências Python do projeto, permitindo instalação rápida com:

```bash
pip install -r requirements.txt
```

#### ⚙️ install.bat
Script de instalação (Windows) para automatizar o setup do ambiente (ex: criação de virtualenv, instalação de dependências, etc.).

#### 📄 README.md
Documento com instruções de uso, informações técnicas e detalhes do projeto.

## 🚀 Como executar

### 1. Instale as dependências

Recomenda-se usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Instale os pacotes:
```bash
pip install -r requirements.txt
```

### 2. Execute o projeto

Usando a webcam:

```bash
python main.py
```

Usando um vídeo (ex: video.mp4):
```bash
python main.py assets/example.mp4
```

Pressione q para encerrar a visualização.

