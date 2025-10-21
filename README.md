vo_hibrido.py: VO monocular y 3D-2D (stereo/depth) con ORB + Essential/PnP.

slam.py: tracking, gestión de keyframes y mapa, bundle adjustment opcional.

visualization.py: visor de trayectoria (Pangolin si existe; Matplotlib si no).

optimizer.py: Bundle Adjustment (si está habilitado).

utils.py: utilidades (proyección, ángulos, etc.).

eval/associate.py y evaluate_ate,py: idk

#en mac 
python3 -m venv .venv
source .venv/bin/activate
#en windows 
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

para slam descargar el dataset, descomprimirlo y crear un acceso 

mkdir -p datasets/tum_rgbd cd datasets/tum_rgbd wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz tar -xvzf rgbd_dataset_freiburg1_desk.tgz

para descomprimir 

para crear ruta 
en mac ln -s /ruta/absoluta/a/rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk
en windows New-Item -ItemType SymbolicLink -Path rgbd_dataset_freiburg1_desk -Target "C:\ruta\absoluta\rgbd_dataset_freiburg1_desk"
si existe esta todo listo ls rgbd_dataset_freiburg1_desk/depth.txt


Hasta ahora tengo este comando bien 
 python vo_hibrido.py --input test1.mp4 --method mono --show
 los demas son solo con el nombre del archivo (sin params) 

 #todo 
 primero probar slam_bb.py 
 luego cambiar la logica de decisor y de slam_Pose para mandar la pose correcta 
 fijarme que anda mejor si slam o vo 