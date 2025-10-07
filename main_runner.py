#!/usr/bin/env python3
"""
CLI unificado para tu proyecto VO-BRIGITTE.
- Mantiene compat con tu flujo (traj.npy -> plot -> triang)
- Permite elegir backend: tu `run_vo` clásico o el nuevo `HybridVO` (mono/stereo/depth)

Ejemplos:
1) Tu pipeline clásico (monocular esencial, sin ventanas):
   python main_runner.py run --input ort.MOV --backend basic --max_frames 150 --no-plot --no-triang

2) Híbrido MONO (Essential) con matches en vivo:
   python main_runner.py run --input 0 --backend hybrid --method mono --show

3) Híbrido STEREO (patrones) + baseline explícita y plot:
   python main_runner.py run --left "left_%06d.png" --right "right_%06d.png" \
       --backend hybrid --method stereo --baseline 0.12 --plot_out img-code/traj.png --show

4) Híbrido DEPTH (RGB + depth_%06d.exr):
   python main_runner.py run --input rgb.mp4 --depth "depth_%06d.exr" --backend hybrid --method depth --show
"""

import argparse
import sys
import numpy as np

# --- Opcionales: si no existen, sigue igual ---
try:
    from vo import run_vo as basic_run_vo
except Exception:
    basic_run_vo = None

try:
    from plot_traj import plot_traj_cli
except Exception:
    plot_traj_cli = None

try:
    from vo_triangulate import triangulate_cli
except Exception:
    triangulate_cli = None

# Híbrido
try:
    from vo_hibrido import VOMethod, HybridVO, open_source, read_frame, estimate_intrinsics
except Exception:
    VOMethod = None
    HybridVO = None
    open_source = None
    read_frame = None
    estimate_intrinsics = None

import cv2


def run_basic(args):
    if basic_run_vo is None:
        raise RuntimeError("No se encontró run_vo en vo.py (backend basic)")
    basic_run_vo(
        args.input,
        out_path=args.traj_npy,
        max_frames=args.max_frames,
        display=args.show,
    )


def run_hybrid(args):
    if HybridVO is None:
        raise RuntimeError("No se encontró vo_hibrido.HybridVO (backend hybrid)")

    # Resolver método
    method_map = {"mono": VOMethod.MONO_2D2D, "stereo": VOMethod.STEREO_3D2D, "depth": VOMethod.DEPTH_3D2D}
    method = method_map[args.method]

    # Abrir fuentes
    if method == VOMethod.MONO_2D2D:
        left_handle, left_kind = open_source(args.input or args.left or "0")
        right_handle, right_kind = (None, None)
    elif method == VOMethod.STEREO_3D2D:
        if not args.left or not args.right:
            raise RuntimeError("Stereo requiere --left y --right")
        left_handle, left_kind = open_source(args.left)
        right_handle, right_kind = open_source(args.right)
    else:  # DEPTH
        left_handle, left_kind = open_source(args.input or args.left or "0")
        if not args.depth:
            raise RuntimeError("Depth mode requiere --depth")
        right_handle, right_kind = (None, None)

    # Primer frame para K
    ok0, frame0 = read_frame(left_handle, left_kind, 0)
    if not ok0:
        raise RuntimeError("No se pudo leer el primer frame")
    K = estimate_intrinsics(frame0.shape, args.fx, args.fy, args.cx, args.cy)

    vo = HybridVO(
        K,
        baseline=args.baseline,
        method=method,
        show=args.show,
        min_matches=args.min_matches,
        ratio_thresh=args.ratio,
        dist_thresh=args.dist,
        use_homography=(not args.no_homography),
        ransac_thresh=args.ransac_thresh,
        save_txt=args.poses_txt,
    )

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    idx = 1
    depth_prev = None

    # Preparar depth inicial según modo
    if method == VOMethod.STEREO_3D2D:
        okr0, frame0_r = read_frame(right_handle, right_kind, 0)
        if not okr0:
            raise RuntimeError("No se pudo leer el primer frame derecho")
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)
        disp0 = sgbm.compute(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame0_r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
        # usar la función privada del objeto para mantener consistencia
        depth_prev = vo._depth_from_disparity(disp0)
    elif method == VOMethod.DEPTH_3D2D:
        dpath0 = args.depth % 0 if "%" in args.depth else args.depth
        depth_prev = cv2.imread(dpath0, cv2.IMREAD_UNCHANGED)
        if depth_prev is None:
            raise RuntimeError("No se pudo leer depth inicial")
        depth_prev = depth_prev.astype(np.float32)

    # Loop
    frames = 1
    while args.max_frames is None or frames < args.max_frames:
        ok, frame = read_frame(left_handle, left_kind, idx)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == VOMethod.MONO_2D2D:
            vo.step_mono(prev_gray, gray, ts=float(frames))
        else:
            if method == VOMethod.STEREO_3D2D:
                okr, frame_r = read_frame(right_handle, right_kind, idx)
                if not okr:
                    break
                sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)
                disp = sgbm.compute(cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY if prev_gray.ndim==3 else prev_gray), cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
                depth_prev = vo._depth_from_disparity(disp)
            else:  # DEPTH
                dpath = args.depth % (idx - 1) if "%" in args.depth else args.depth
                depth_prev = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
                if depth_prev is None:
                    break
                depth_prev = depth_prev.astype(np.float32)
            vo.step_pnp(prev_gray, gray, depth_prev, ts=float(frames))

        if args.show:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        prev_gray = gray
        idx += 1
        frames += 1

    if args.show:
        cv2.destroyAllWindows()

    traj = np.vstack(vo.traj).astype(np.float32)
    np.save(args.traj_npy, traj)


def main():
    ap = argparse.ArgumentParser("Runner VO-BRIGITTE (unificado)")
    ap.add_argument("cmd", choices=["run"], help="run")

    # Backends
    ap.add_argument("--backend", choices=["basic", "hybrid"], default="basic")

    # Fuentes (mono / stereo / depth)
    ap.add_argument("--input", help="Video/ruta/índice (mono o depth)")
    ap.add_argument("--left", help="Patrón/Video cámara izquierda (stereo)")
    ap.add_argument("--right", help="Patrón/Video cámara derecha (stereo)")
    ap.add_argument("--depth", help="Patrón de depth (p.ej. depth_%06d.exr)")

    # Modo híbrido
    ap.add_argument("--method", choices=["mono", "stereo", "depth"], default="mono")
    ap.add_argument("--baseline", type=float, default=0.1)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--dist", type=float, default=50.0)
    ap.add_argument("--no_homography", action="store_true")
    ap.add_argument("--ransac_thresh", type=float, default=1.0)

    # General
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-triang", action="store_true")

    # Salidas
    ap.add_argument("--traj_npy", default="trajectory.npy")
    ap.add_argument("--plot_out", default="img-code/traj.png")
    ap.add_argument("--poses_txt", default="poses.txt")

    args = ap.parse_args()

    # 1) VO -> trajectory.npy
    print("▶ VO…")
    if args.backend == "basic":
        if not args.input:
            print("⚠️ basic backend necesita --input (video o cámara)")
            sys.exit(1)
        run_basic(args)
    else:
        run_hybrid(args)

    # 2) Plot
    img = None
    if not args.no_plot and plot_traj_cli is not None:
        try:
            print("▶ Plot…")
            img = plot_traj_cli(args.traj_npy, args.plot_out)
        except Exception as e:
            print(f"Plot falló: {e}")

    # 3) Triangulación (tu CLI)
    ply = None
    if not args.no_triang and triangulate_cli is not None and args.input:
        try:
            print("▶ Triangulación…")
            ply = triangulate_cli(args.input, "points.ply", max_frames=args.max_frames)
        except Exception as e:
            print(f"Triangulación falló: {e}")

    print("✅ DONE")
    print({"trajectory": args.traj_npy, "plot": img, "points": ply})


if __name__ == "__main__":
    main()
