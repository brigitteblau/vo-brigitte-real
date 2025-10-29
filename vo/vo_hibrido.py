#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
import csv
import time
import os


@dataclass
class VOResult:
    """Salida clara para el pipeline."""
    e_lat: float          # error lateral normalizado [-1..1] (izq + / der -)
    e_head_deg: float     # error de heading en grados (+ izquierda / - derecha)
    conf: float           # confianza [0..1]
    overlay: Optional[np.ndarray] = None  # frame anotado si visualize=True


class VOHibrido:
    """
    VO liviano basado en líneas (lane-only).
    - Calcula error lateral y de heading respecto al centro de imagen.
    - No usa ArUco ni SLAM pesado. Ideal para WCADS + EKF + PID.

    Inputs:
      - frame: np.ndarray BGR (OpenCV), resolución cualquiera (ideal 640x480)
    Outputs:
      - VOResult(e_lat, e_head_deg, conf, overlay)
    """

    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
        visualize: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.K = K
        self.dist = dist
        self.visualize = visualize
        self.cfg = {
            # ROI inferior (porcentaje de alto)
            "roi_y0": 0.55,   # desde 55% del alto hasta abajo
            # preprocesado
            "blur_ksize": 5,
            "canny1": 60,
            "canny2": 150,
            # Hough
            "hough_rho": 1.0,
            "hough_theta_deg": 1.0,
            "hough_thresh": 30,
            "hough_min_len": 30,
            "hough_max_gap": 10,
            # post
            "min_lines_conf": 10,    # cuantas líneas suman confianza razonable
            "max_head_for_conf": 45, # |heading| razonable para dar confianza alta
        }
        if cfg:
            self.cfg.update(cfg)

    # ------------- API principal -------------
    def step(self, frame: np.ndarray) -> VOResult:
        """
        Procesa un frame BGR y devuelve errores + confianza + overlay opcional.
        """
        if frame is None or frame.size == 0:
            return VOResult(0.0, 0.0, 0.0, None)

        # 1) ROI inferior
        h, w = frame.shape[:2]
        y0 = int(self.cfg["roi_y0"] * h)
        roi = frame[y0:h, :]

        # 2) Preprocess (gris + blur + canny)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.cfg["blur_ksize"] >= 3 and self.cfg["blur_ksize"] % 2 == 1:
            gray = cv2.GaussianBlur(gray, (self.cfg["blur_ksize"], self.cfg["blur_ksize"]), 0)
        edges = cv2.Canny(gray, self.cfg["canny1"], self.cfg["canny2"])

        # 3) HoughLinesP para segmentos
        lines = cv2.HoughLinesP(
            edges,
            rho=self.cfg["hough_rho"],
            theta=np.deg2rad(self.cfg["hough_theta_deg"]),
            threshold=self.cfg["hough_thresh"],
            minLineLength=self.cfg["hough_min_len"],
            maxLineGap=self.cfg["hough_max_gap"],
        )

        # Si no hay líneas → devolver neutro con baja confianza
        if lines is None or len(lines) == 0:
            overlay = self._build_overlay(frame, y0, None, None, 0.0) if self.visualize else None
            return VOResult(0.0, 0.0, 0.05, overlay)

        # 4) Calcular heading y lateral
        head_deg, e_lat_norm, conf = self._compute_errors_and_conf(lines, w, y0)

        # 5) Overlay
        overlay = None
        if self.visualize:
            overlay = self._build_overlay(frame, y0, lines, (head_deg, e_lat_norm, conf), conf)

        return VOResult(e_lat=e_lat_norm, e_head_deg=head_deg, conf=conf, overlay=overlay)

    # ------------- Helpers internos -------------
    def _compute_errors_and_conf(self, lines, width: int, roi_y0: int) -> Tuple[float, float, float]:
        """
        A partir de segmentos Hough, calcula:
        - head_deg: heading promedio (convención: + izquierda / - derecha)
        - e_lat_norm: offset lateral normalizado respecto al centro de imagen
        - conf: confianza 0..1 basada en #líneas y coherencia angular
        """
        angles = []
        xs = []
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            # Angulo en grados (en ROI, y hacia abajo positivo)
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Convertimos convención a "girar a izq = positivo"
            head = -ang
            angles.append(head)
            xs.extend([x1, x2])

        if len(angles) == 0:
            return 0.0, 0.0, 0.05

        head_deg = float(np.mean(angles))
        lane_x = float(np.mean(xs))
        center_x = width / 2.0
        e_lat_px = lane_x - center_x
        # Normalizado para que +-width/2 -> +-1
        e_lat_norm = float(e_lat_px / (width / 2.0))

        # Confianza:
        #   - más líneas -> más confianza
        #   - penalizamos si el heading es extremo
        n = len(angles)
        n_norm = min(1.0, n / float(self.cfg["min_lines_conf"]))
        head_penalty = max(0.0, 1.0 - (abs(head_deg) / self.cfg["max_head_for_conf"]))
        # coherencia angular (menor varianza = mejor)
        ang_std = float(np.std(angles)) if n >= 2 else 0.0
        coh = max(0.0, 1.0 - min(1.0, ang_std / 20.0))  # 20° std ~ 0 de coherencia

        conf = float(np.clip(0.2 * n_norm + 0.5 * head_penalty + 0.3 * coh, 0.0, 1.0))
        return head_deg, e_lat_norm, conf

    def _build_overlay(
        self,
        frame_bgr: np.ndarray,
        roi_y0: int,
        lines,
        metrics: Optional[Tuple[float, float, float]],
        conf: float,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        # ROI box
        cv2.rectangle(out, (0, roi_y0), (w, h), (40, 40, 40), 1)
        # Dibujar líneas
        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, l)
                cv2.line(out, (x1, y1 + roi_y0), (x2, y2 + roi_y0), (0, 255, 255), 2)
        # Texto
        if metrics is not None:
            head_deg, e_lat_norm, conf_val = metrics
            cv2.putText(out, f"e_lat={e_lat_norm:.3f}  e_head={head_deg:.1f} deg  conf={conf_val:.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(out, f"sin lineas  conf={conf:.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        return out


# ------------- CLI para prueba rápida con tu video -------------
def run_video(input_path: str, output_csv: Optional[str] = None, show: bool = False, cfg: Optional[Dict[str, Any]] = None):
    """
    Reproduce un video, corre VO por frame y (opcional) guarda CSV de resultados.
    CSV columnas: timestamp,e_lat,e_head_deg,conf
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    vo = VOHibrido(visualize=show, cfg=cfg)
    writer = None
    if output_csv:
        writer = csv.writer(open(output_csv, "w", newline="", encoding="utf-8"))
        writer.writerow(["timestamp", "e_lat", "e_head_deg", "conf"])

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = vo.step(frame)
        ts = time.time() - t0
        if writer:
            writer.writerow([f"{ts:.6f}", f"{res.e_lat:.6f}", f"{res.e_head_deg:.3f}", f"{res.conf:.3f}"])

        if show:
            cv2.imshow("VO Hibrido (lane-only)", res.overlay if res.overlay is not None else frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ejemplo CLI: python -m vo.vo_hibrido --input tu_video.mp4 --show --csv out.csv
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta a video de prueba")
    ap.add_argument("--csv", default=None, help="Guardar CSV con e_lat, e_head_deg, conf")
    ap.add_argument("--show", action="store_true", help="Mostrar overlay")
    args = ap.parse_args()
    run_video(args.input, output_csv=args.csv, show=args.show)
