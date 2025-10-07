hay que instalar el dataset antes de usarlo el dataset tum que no se puede subir al repo 

Dataset externo

GitHub no permite subir archivos mayores a 100 MB, por eso el dataset original
datasets/datasets/rgbd_dataset_freiburg1_desk.tgz no se incluye en el repo.

Descargalo desde:
👉 TUM RGB-D Dataset – freiburg1_desk

Y colocalo en la carpeta:

datasets/datasets/


Estructura esperada:

vo-brigitte/
 ├── main_runner.py
 ├── vo_hibrido.py
 ├── optimizer.py
 └── datasets/
      └── datasets/
           └── rgbd_dataset_freiburg1_desk.tgz

Perfecto 💪 te dejo cómo documentar eso todo en el README, con el comando que usaste para instalar el repo y el fix del push grande.

🧰 Instalación local
# 1️⃣ Cloná el repo
git clone https://github.com/brigitteblau/vo-brigitte.git
cd vo-brigitte

# 2️⃣ Creá y activá el entorno virtual
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# o en Windows:
# .venv\Scripts\activate

# 3️⃣ Instalá dependencias
pip install -r requirements.txt


⚠️ Si usás g2o-python y falla en macOS ARM, podés correr:

pip install -U g2o-python --only-binary=:all:


o simplemente usar el fallback de OpenCV (solvePnPRansac)
