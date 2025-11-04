import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Parámetros para detección de características
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parámetros para flujo óptico Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Leer primer frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear máscara para dibujar
mask = np.zeros_like(old_frame)

print("Detector iniciado. Muévete con la cámara. Presiona 'q' para salir.")
print("- Camina ADELANTE: los puntos se expanden desde el centro")
print("- Camina ATRAS: los puntos se contraen hacia el centro")
print("- Gira DERECHA/IZQUIERDA: los puntos se mueven horizontalmente")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calcular flujo óptico
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            # Seleccionar buenos puntos
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) > 10:
                # Calcular vectores de movimiento
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                vectors = good_new - good_old
                positions = good_old
                
                # Calcular distancias desde el centro
                distances_old = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
                distances_new = np.sqrt((good_new[:, 0] - center_x)**2 + (good_new[:, 1] - center_y)**2)
                
                # Cambio en distancia (expansión/contracción)
                radial_change = np.mean(distances_new - distances_old)
                
                # Movimiento horizontal promedio
                horizontal_movement = np.mean(vectors[:, 0])
                
                # Movimiento vertical promedio
                vertical_movement = np.mean(vectors[:, 1])
                
                # Determinar dirección
                direccion = ""
                if abs(radial_change) > abs(horizontal_movement) * 0.5:
                    if radial_change > 1.5:
                        direccion = "ADELANTE"
                    elif radial_change < -1.5:
                        direccion = "ATRAS"
                
                if abs(horizontal_movement) > 2:
                    if horizontal_movement > 0:
                        direccion = "IZQUIERDA" if direccion == "" else direccion + " + IZQUIERDA"
                    else:
                        direccion = "DERECHA" if direccion == "" else direccion + " + DERECHA"
                
                if direccion:
                    print(f"Movimiento: {direccion}")
                
                # Dibujar vectores
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)
                
                # Mostrar dirección en pantalla
                if direccion:
                    cv2.putText(frame, direccion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (0, 255, 0), 3)
            
            # Actualizar puntos anteriores
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    
    # Recalcular puntos cada 30 frames
    frame_count += 1
    if frame_count % 30 == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
    
    # Combinar frame con máscara
    img = cv2.add(frame, mask)
    
    cv2.imshow('Detector de Movimiento - Flujo Optico', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()