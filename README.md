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
1️⃣ Descargar el dataset TUM RGB-D

Elegí uno de los más livianos para probar, por ejemplo freiburg1_desk:

mkdir -p datasets/tum_rgbd
cd datasets/tum_rgbd
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz


👉 Esto te deja una carpeta:

datasets/tum_rgbd/rgbd_dataset_freiburg1_desk/


con los archivos rgb.txt, depth.txt, etc.

(Si estás en macOS y no tenés wget, podés usar curl):

curl -O https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz

⚙️ 2️⃣ Crear un acceso directo (enlace simbólico)

Así no tenés que mover el dataset dentro de tu proyecto.

Supongamos que tu proyecto está en:

/Users/brigu/Desktop/vo-brigitte-real/


Y descargaste el dataset en:

/Users/brigu/Desktop/datasets/tum_rgbd/rgbd_dataset_freiburg1_desk/


Desde la carpeta del proyecto (vo-brigitte-real), ejecutá:

ln -s /Users/brigu/Desktop/datasets/tum_rgbd/rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk


💡 Esto crea un “acceso directo” dentro del proyecto, como si el dataset estuviera ahí.

🧭 3️⃣ Verificá que existe depth.txt
ls rgbd_dataset_freiburg1_desk/depth.txt


Si aparece, ya está OK ✅
Si no, asegurate de que el tar -xvzf realmente descomprimió los archivos