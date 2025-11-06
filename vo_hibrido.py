import argparse
import os
import sys
import time
import cv2
import numpy as np
from enum import Enum
from visualization import init_traj_view, traj_update_from_pose, set_3d_view_params,  save_trajectory_npy
from control.decisor import Decisor
from optimizer import BundleAdjustment

USE_G2O = False
try:
    import g2o 
    USE_G2O = True
except Exception:
    USE_G2O = False

fx = 640.0     # focal en píxeles eje X
fy = 640.0     # focal en píxeles eje Y
cx = 640.0     # centro óptico en X (mitad del ancho)
cy = 360.0     # centro óptico en Y (mitad de la altura)

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

D = np.zeros((5, 1), dtype=np.float64)


class VOMethod(Enum):
    MONO_2D2D = 1  
    STEREO_3D2D = 2 
    DEPTH_3D2D = 3 

import cv2



# --- Wrapper para usar Picamera2 como si fuera cv2.VideoCapture
class PiCamCapture:
    def __init__(self, size=(640, 480), fps=30, fourcc="XRGB8888"):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        # Configuración de video estable para CV (formato 8-bit)
        self.picam2.configure(
            self.picam2.create_video_configuration(
                main={"format": fourcc, "size": size},
                controls={"FrameDurationLimits": (int(1e6/fps), int(1e6/fps))}
            )
        )
        self.picam2.start()

    def isOpened(self):
        return True

    def read(self):
        frame = self.picam2.capture_array()
        # Devuelve (ret, frame) como OpenCV
        return True, frame

    def release(self):
        self.picam2.stop()

def _try_v4l2(index=0, width=None, height=None, fps=None):
    # Intenta abrir /dev/video{index} con V4L2
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap if cap.isOpened() else None

def _try_gstreamer(width=640, height=480, fps=30):
    # Solo si tu OpenCV fue compilado con GStreamer
    pipeline = (
        f"libcamerasrc ! video/x-raw, width={width}, height={height}, "
        f"framerate={fps}/1 ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap if cap.isOpened() else None

def _try_picamera2(width=640, height=480, fps=30):
    try:
        cap = PiCamCapture(size=(width, height), fps=fps)
        return cap
    except Exception:
        return None

def open_source(src, width=640, height=480, fps=30):
    """
    Abre fuente de imágenes para tu VO:
      - '0' o '1' => intenta v4l2 (/dev/videoX) y si falla cae a Picamera2
      - 'raspi'/'pi'/'picam'/'libcamera' => abre Picamera2 directo
      - patrón con % o * => secuencia de imágenes
      - otra string => ruta a video
    Retorna (cap_or_path, kind) donde kind in {"video","images","raspi"}.
    """
    if src is None:
        return None, None

    # Opción explícita: forzá la cámara Pi via libcamera
    if isinstance(src, str) and src.lower() in {"raspi", "pi", "picam", "libcamera"}:
        cap = _try_picamera2(width, height, fps)
        if cap: return cap, "raspi"
        # fallback por si tu OpenCV tiene gstreamer
        cap = _try_gstreamer(width, height, fps)
        if cap: return cap, "video"
        return None, None

    # Índice de cámara "0", "1", etc.
    if isinstance(src, str) and src.isdigit():
        cap = _try_v4l2(int(src), width, height, fps)
        if cap: return cap, "video"
        # fallback a Picamera2 si /dev/videoX no existe
        cap = _try_picamera2(width, height, fps)
        if cap: return cap, "raspi"
        # último intento: gstreamer
        cap = _try_gstreamer(width, height, fps)
        if cap: return cap, "video"
        return None, None

    # Patrón de imágenes
    if isinstance(src, str) and ("%" in src or "*" in src):
        return src, "images"

    # Ruta a archivo de video
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        return cap, "video"

    # Si todo falla y estás en Pi, intentá Picamera2
    cap = _try_picamera2(width, height, fps)
    if cap: return cap, "raspi"

    return None, None


def read_frame(handle, kind, idx):
    """Lee frame i-ésimo ya sea de video o patrón de imágenes."""
    if handle is None:
        return False, None
    if kind == "video":
        ok, frame = handle.read()
        return ok, frame
    else:
        path = handle % idx
        if not os.path.exists(path):
            return False, None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img is not None, img


def estimate_intrinsics(frame_shape, fx=None, fy=None, cx=None, cy=None):
    h, w = frame_shape[:2]
    if fx is None or fy is None:
        fx = fy = 0.9 * w
    if cx is None or cy is None:
        cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


def to_quat(R):
    """Convierte R(3x3) a cuaternión (x,y,z,w)."""
    q = np.empty(4, dtype=np.float64)
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (m[2, 1] - m[1, 2]) / s
        q[1] = (m[0, 2] - m[2, 0]) / s
        q[2] = (m[1, 0] - m[0, 1]) / s
    else:
        i = np.argmax([m[0, 0], m[1, 1], m[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[0] = 0.25 * s
            q[1] = (m[0, 1] + m[1, 0]) / s
            q[2] = (m[0, 2] + m[2, 0]) / s
            q[3] = (m[2, 1] - m[1, 2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[0] = (m[0, 1] + m[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (m[1, 2] + m[2, 1]) / s
            q[3] = (m[0, 2] - m[2, 0]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[0] = (m[0, 2] + m[2, 0]) / s
            q[1] = (m[1, 2] + m[2, 1]) / s
            q[2] = 0.25 * s
            q[3] = (m[1, 0] - m[0, 1]) / s
    return q  


def yaw_from_R(R):
    """Devuelve yaw (rad) de la rotación (convención Z yaw, XYZ)."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class HybridVOcd:
    def __init__(self, K, baseline=0.1, method=VOMethod.MONO_2D2D,
                 show=True, min_matches=30, ratio_thresh=0.7, dist_thresh=50.0,
                 use_homography=True, ransac_thresh=1.0, save_txt=None):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.baseline = baseline
        self.method = method
        self.show = show
        self.min_matches = min_matches
        self.ratio_thresh = ratio_thresh
        self.dist_thresh = dist_thresh
        self.use_homography = use_homography
        self.ransac_thresh = ransac_thresh
        self.save_txt = save_txt

        self.R_cum = np.eye(3)
        self.t_cum = np.zeros((3, 1))
        self.traj = [self.t_cum.ravel().copy()]

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)

        self.decisor = Decisor()
        self.on_command = lambda cmd: print(f"[CMD] {cmd}")

        if self.save_txt:
            with open(self.save_txt, "w") as f:
                f.write("# timestamp x y z qx qy qz qw\n")

        # ------------------ SLAM: BA + buffers ------------------
        self.ba = None
        try:
            from optimizer import BundleAdjustment
            self.ba = BundleAdjustment(fx=self.K[0,0], cx=self.K[0,2], cy=self.K[1,2], verbose=False)
        except Exception:
            self.ba = None  # sigue todo sin BA

        # Buffers de mapa
        self.keyframes = []          # lista de objetos con .pose (4x4 T_cw), .kpts ([(u,v),...]), .map_point_ids
        self.map_points = []         # objetos con .position (3,), .observed_keyframe_ids
        self.last_kf_id = None
        self.last_kf_kps = None
        self.last_kf_desc = None

        # Heurísticas KF/BA
        self.kf_every = 5
        self.kf_min_inliers = 60
        self.kf_parallax_deg = 1.5

    # ---------- Helpers compatibles ----------
    def _convert_grayscale(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.ndim == 2:
            return img_bgr
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def _compute_orb(self, gray: np.ndarray):
        kpts, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kpts) == 0:
            return [], np.empty((0, 32), dtype=np.uint8)
        return kpts, desc

    def _project_2d_kpts_to_3d(self, depth_img: np.ndarray, kpts_xy: np.ndarray) -> np.ndarray:
        fx = self.K[0, 0]; fy = self.K[1, 1]; cx = self.K[0, 2]; cy = self.K[1, 2]
        N = len(kpts_xy)
        pts3d = np.full((N, 3), np.nan, dtype=float)
        h, w = depth_img.shape[:2]
        u = np.clip(np.round(kpts_xy[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(kpts_xy[:, 1]).astype(int), 0, h - 1)
        z = depth_img[v, u].astype(float)
        valid = np.isfinite(z) & (z > 0)
        x = (u.astype(float) - cx) * (z / fx)
        y = (v.astype(float) - cy) * (z / fy)
        pts3d[valid, 0] = x[valid]; pts3d[valid, 1] = y[valid]; pts3d[valid, 2] = z[valid]
        return pts3d

    def _get_matches(self, prev_kpts_xy, curr_kpts_xy, prev_desc, curr_desc):
        if prev_desc is None or curr_desc is None or len(prev_desc) == 0 or len(curr_desc) == 0:
            return []
        knn = self.bf.knnMatch(prev_desc, curr_desc, k=2)
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def _minimize_reprojection_error(self, p2d_pix: np.ndarray, p3d_cam_pred: np.ndarray) -> np.ndarray:
        valid = np.isfinite(p3d_cam_pred).all(axis=1)
        obj = p3d_cam_pred[valid]
        img = p2d_pix[valid]
        if len(obj) < 6:
            return np.eye(4, dtype=float)
        obj = obj.astype(np.float32)
        img = img.astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)[:3]
        if not ok:
            return np.eye(4, dtype=float)
        Rm, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rm
        T[:3, 3] = tvec.ravel()
        return T

    # ---------- Matching robusto ----------
    def _get_matches_kp(self, k1, k2, d1, d2):
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            return []
        try:
            raw = self.flann.knnMatch(d1, d2, k=2)
        except Exception:
            return []
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance and m.distance < self.dist_thresh:
                good.append(m)
        if not self.use_homography or len(good) < 4:
            return good
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            return good
        return [m for m, msk in zip(good, mask) if msk[0] == 1]

    def _depth_from_disparity(self, disp):
        disp = disp.astype(np.float32)
        disp[disp <= 0] = np.nan
        return (self.K[0, 0] * self.baseline) / disp

    def _project_2d_to_3d(self, depth, pts2d):
        if pts2d.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        u = pts2d[:, 0].astype(np.int32)
        v = pts2d[:, 1].astype(np.int32)
        valid = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        u = u[valid]; v = v[valid]
        z = depth[v, u]
        keep = ~np.isnan(z)
        u = u[keep]; v = v[keep]; z = z[keep]
        pts = np.stack([u * z, v * z, z], axis=1).astype(np.float64)
        pts = (self.K_inv @ pts.T).T  # (N,3)
        return pts

    def _solve_pnp(self, pts3d, pts2d):
        if USE_G2O and pts3d.shape[0] >= 6:
            try:
                optimizer = g2o.SarseOptimizer()  # typo intencional evitado en tu versión original
            except Exception:
                pass
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64), pts2d.astype(np.float64), self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0, iterationsCount=100
        )
        if not ok:
            return None, None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        return R, t

    # ------------------- NUEVO: helpers SLAM -------------------
    def _current_Tcw(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R_cum
        T[:3, 3:4] = self.t_cum
        return T

    def _parallax_deg(self, pts1, pts2):
        # normaliza con K y mide ángulo mediano
        ph1 = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3).T
        ph2 = cv2.convertPointsToHomogeneous(pts2).reshape(-1,3).T
        n1 = self.K_inv @ ph1; n1 = (n1[:2]/n1[2]).T
        n2 = self.K_inv @ ph2; n2 = (n2[:2]/n2[2]).T
        num = np.sum(n1*n2, axis=1)
        den = np.linalg.norm(n1,axis=1)*np.linalg.norm(n2,axis=1) + 1e-9
        ang = np.degrees(np.arccos(np.clip(num/den, -1.0, 1.0)))
        return float(np.median(ang))

    def _triangulate_between(self, Tcw_ref, Tcw_cur, pts_ref, pts_cur):
        # Proyecciones P = K [R|t] (cámara <- mundo)
        Rr, tr = Tcw_ref[:3,:3], Tcw_ref[:3,3]
        Rc, tc = Tcw_cur[:3,:3], Tcw_cur[:3,3]
        Pr = self.K @ np.hstack([Rr, tr.reshape(3,1)])
        Pc = self.K @ np.hstack([Rc, tc.reshape(3,1)])

        pr = pts_ref.astype(np.float64).reshape(-1,1,2)
        pc = pts_cur.astype(np.float64).reshape(-1,1,2)
        X_h = cv2.triangulatePoints(Pr, Pc, pr, pc)  # 4xN
        X = (X_h[:3,:] / X_h[3,:]).T  # Nx3

        # chequear z>0 en ambos
        def _depth(Tcw, Xw):
            Xc = (Tcw[:3,:3] @ Xw.T + Tcw[:3,3:4]).T
            return Xc[:,2]
        z_ref = _depth(Tcw_ref, X)
        z_cur = _depth(Tcw_cur, X)
        valid = (z_ref > 0.1) & (z_cur > 0.1) & np.isfinite(X).all(axis=1)
        return X, valid

    # ---------- Control: emitir comando ----------
    def _emit_cmd_from_pose(self):
        x, y, _ = self.t_cum.ravel()
        yaw = yaw_from_R(self.R_cum)
        cmd, should_send = self.decisor.decide(x, y, yaw)
        if should_send:
            try:
                self.on_command(cmd)
            except Exception as e:
                print(f"[WARN] fallo al enviar comando: {e}")

    # ---------- Pasos VO ----------
    def step_mono(self, prev_gray, gray, ts):
        # ORB en prev/cur
        k1, d1 = self.orb.detectAndCompute(prev_gray, None)
        k2, d2 = self.orb.detectAndCompute(gray, None)
        good_12 = self._get_matches_kp(k1, k2, d1, d2)
        if len(good_12) < self.min_matches:
            return False

        pts1 = np.float32([k1[m.queryIdx].pt for m in good_12])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good_12])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=self.ransac_thresh)
        if E is None or mask is None:
            return False

        in1 = pts1[mask.ravel() == 1]
        in2 = pts2[mask.ravel() == 1]
        if len(in1) < 5:
            return False

        _, R, t, _ = cv2.recoverPose(E, in1, in2, self.K)

        # Acumular (coords mundo)
        t_step = (self.R_cum @ t)
        self.t_cum = self.t_cum + t_step
        self.R_cum = R @ self.R_cum
        self.traj.append(self.t_cum.ravel().copy())

        if self.show:
            draw = cv2.drawMatches(prev_gray, k1, gray, k2, good_12[:100], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches (MONO)", draw)
            traj_update_from_pose(self.t_cum.ravel().copy())

        # -------- SLAM: inicialización de primer KF si hace falta --------
        if self.last_kf_id is None:
            kpts_px = np.array([kp.pt for kp in k1], dtype=np.float64)
            self.keyframes.append(
                type("KF", (), {})()
            )
            self.keyframes[0].pose = self._current_Tcw().copy()
            self.keyframes[0].kpts = kpts_px.tolist()
            self.keyframes[0].map_point_ids = [None]*len(k1)
            self.last_kf_id = 0
            self.last_kf_kps = k1
            self.last_kf_desc = d1

        # -------- SLAM: promover KF y triangulación con último KF --------
        promote = False
        pts_ref_i = None; pts_cur_i = None; idx_ref_i = None; idx_cur_i = None

        if self.last_kf_desc is not None:
            # matches último KF vs frame actual
            good_kf = self._get_matches_kp(self.last_kf_kps, k2, self.last_kf_desc, d2)
            if len(good_kf) >= self.kf_min_inliers:
                pts_ref = np.float32([self.last_kf_kps[m.queryIdx].pt for m in good_kf])
                pts_cur = np.float32([k2[m.trainIdx].pt for m in good_kf])

                E2, mask2 = cv2.findEssentialMat(pts_ref, pts_cur, self.K, method=cv2.RANSAC,
                                                 prob=0.999, threshold=self.ransac_thresh)
                if E2 is not None and mask2 is not None:
                    inl2 = mask2.ravel() == 1
                    pts_ref_i = pts_ref[inl2]
                    pts_cur_i = pts_cur[inl2]
                    if len(pts_ref_i) >= self.kf_min_inliers:
                        par = self._parallax_deg(pts_ref_i, pts_cur_i)
                        promote = (par > self.kf_parallax_deg)
                        # guardo índices originales para mapear a kpts
                        idx_ref_i = np.array([good_kf[i].queryIdx for i, b in enumerate(inl2) if b], dtype=int)
                        idx_cur_i = np.array([good_kf[i].trainIdx for i, b in enumerate(inl2) if b], dtype=int)

        if promote:
            # crear KF actual
            k2_all, d2_all = k2, d2
            kpts_px_cur = np.array([kp.pt for kp in k2_all], dtype=np.float64)

            self.keyframes.append(type("KF", (), {})())
            cur_kf = self.keyframes[-1]
            cur_kf.pose = self._current_Tcw().copy()
            cur_kf.kpts = kpts_px_cur.tolist()
            cur_kf.map_point_ids = [None]*len(k2_all)
            cur_kf_id = len(self.keyframes) - 1

            # Triangulación
            X, valid = self._triangulate_between(self.keyframes[self.last_kf_id].pose,
                                                 cur_kf.pose,
                                                 pts_ref_i, pts_cur_i)

            # Crear MPs y asociar a índices exactos de kpts (ref y cur)
            for j, v in enumerate(valid):
                if not v:
                    continue
                mp = type("MP", (), {})()
                mp.position = X[j].astype(np.float64)
                mp.observed_keyframe_ids = [self.last_kf_id, cur_kf_id]
                self.map_points.append(mp)
                mp_id = len(self.map_points) - 1

                # asignar a listas map_point_ids usando índices de kpts
                self.keyframes[self.last_kf_id].map_point_ids[idx_ref_i[j]] = mp_id
                cur_kf.map_point_ids[idx_cur_i[j]] = mp_id

            # actualizar referencia KF
            self.last_kf_id = cur_kf_id
            self.last_kf_kps = k2_all
            self.last_kf_desc = d2_all

            # BA periódico
            if self.ba and (len(self.keyframes) % self.kf_every == 0):
                self.keyframes, self.map_points = self.ba.optimize(
                    self.keyframes, self.map_points, num_iterations=20, sigma_px=1.5
                )

        self._emit_cmd_from_pose()
        self._maybe_save_pose(ts)
        return True

    def step_pnp(self, prev_gray, gray, depth_prev, ts):
        k1, d1 = self.orb.detectAndCompute(prev_gray, None)
        k2, d2 = self.orb.detectAndCompute(gray, None)
        good = self._get_matches_kp(k1, k2, d1, d2)
        if len(good) < self.min_matches:
            return False

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        pts3d = self._project_2d_to_3d(depth_prev, pts1)
        if pts3d.shape[0] < 6:
            return False

        valid_mask = ~np.isnan(pts3d).any(axis=1)
        pts3d = pts3d[valid_mask]
        pts2d = pts2[valid_mask]
        if pts3d.shape[0] < 6:
            return False

        R, t = self._solve_pnp(pts3d, pts2d)
        if R is None:
            return False

        t_step = (self.R_cum @ t)
        self.t_cum = self.t_cum + t_step
        self.R_cum = R @ self.R_cum
        self.traj.append(self.t_cum.ravel().copy())

        if self.show:
            draw = cv2.drawMatches(prev_gray, k1, gray, k2, good[:100], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches (PnP)", draw)
            traj_update_from_pose(self.t_cum.ravel().copy())

        self._emit_cmd_from_pose()
        self._maybe_save_pose(ts)
        return True

    def _maybe_save_pose(self, ts):
        if not self.save_txt:
            return
        Rm = self.R_cum
        t = self.t_cum.ravel()
        qx, qy, qz, qw = to_quat(Rm)
        with open(self.save_txt, "a") as f:
            f.write(f"{ts:.6f} {t[0]} {t[1]} {t[2]} {qx} {qy} {qz} {qw}\n")


# ------------- Main loop -----------------

def main():
    ap = argparse.ArgumentParser(description="VO Híbrido (Mono/Stereo/Depth) — ORB + E/PnP")
    ap.add_argument("--input", help="Video/índice/patrón imágenes para cámara izquierda o monocular", default=None)
    ap.add_argument("--left", help="Patrón/Video cámara izquierda (stereo)", default=None)
    ap.add_argument("--right", help="Patrón/Video cámara derecha (stereo)", default=None)
    ap.add_argument("--depth", help="Patrón mapa de profundidad (si existe)", default=None)
    ap.add_argument("--method", choices=["mono", "stereo", "depth"], default="mono")
    ap.add_argument("--baseline", type=float, default=0.1)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--traj_npy", default="trajectory.npy")
    ap.add_argument("--poses_txt", default="poses.txt")
    ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--dist", type=float, default=50.0)
    ap.add_argument("--no_homography", action="store_true")
    ap.add_argument("--ransac_thresh", type=float, default=1.0)
    ap.add_argument("--udp", help="ip:puerto para enviar comandos por UDP (opcional)", default=None)
    args = ap.parse_args()

    method = {
        "mono": VOMethod.MONO_2D2D,
        "stereo": VOMethod.STEREO_3D2D,
        "depth": VOMethod.DEPTH_3D2D,
    }[args.method]

    # Abrir fuentes
    if args.input and (args.left or args.right):
        print("⚠️ Usa --input (mono/depth) o --left/--right (stereo), no ambos.")
        sys.exit(1)

    left_handle, left_kind = (None, None)
    right_handle, right_kind = (None, None)

    if method == VOMethod.MONO_2D2D:
        left_handle, left_kind = open_source(args.input or args.left or "0")
    elif method == VOMethod.STEREO_3D2D:
        left_handle, left_kind = open_source(args.left)
        right_handle, right_kind = open_source(args.right)
        if left_handle is None or right_handle is None:
            print("❌ Stereo requiere --left y --right")
            sys.exit(1)
    else:  # DEPTH
        left_handle, left_kind = open_source(args.input or args.left or "0")
        if args.depth is None:
            print("❌ Depth mode requiere --depth (patrón de archivos .png/.exr)")
            sys.exit(1)

    # Leer primer frame(s) para intrínsecos
    ok0, frame0 = read_frame(left_handle, left_kind, 0)
    if not ok0:
        print("❌ No se pudo leer el primer frame de la izquierda/mono")
        sys.exit(1)
    K = estimate_intrinsics(frame0.shape, args.fx, args.fy, args.cx, args.cy)

    # Crear VO
    vo = HybridVOcd(K, baseline=args.baseline, method=method, show=args.show,
                    min_matches=args.min_matches, ratio_thresh=args.ratio,
                    dist_thresh=args.dist, use_homography=(not args.no_homography),
                    ransac_thresh=args.ransac_thresh, save_txt=args.poses_txt)

    # Inicializar salida UDP si corresponde
    if args.udp:
        import socket
        host, port = args.udp.split(":")
        addr = (host, int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def _send_udp(cmd: str):
            sock.sendto(cmd.encode("utf-8"), addr)
        vo.on_command = _send_udp
        print(f"[INFO] Enviando comandos UDP a {addr}")

    # Inicializar visor si corresponde
    if args.show:
        init_traj_view("Trayectoria (VO)")

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    idx = 1
    depth_prev = None

    # Si stereo, preparar depth del frame 0 (alineando left0-right0)
    if method == VOMethod.STEREO_3D2D:
        okr0, frame0_r = read_frame(right_handle, right_kind, 0)
        if not okr0:
            print("❌ No se pudo leer el primer frame derecho")
            sys.exit(1)
        disp0 = vo.sgbm.compute(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(frame0_r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
        depth_prev = vo._depth_from_disparity(disp0)
        if args.show:
            dv = np.nan_to_num(depth_prev, nan=5.0)
            dv = np.clip(dv, 0.0, 5.0)
            dv = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dv = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO)
            cv2.imshow("Depth", dv)

    # Si depth puro, cargar el depth del frame 0
    if method == VOMethod.DEPTH_3D2D:
        dpath0 = args.depth % 0 if "%" in args.depth else args.depth
        depth_prev = cv2.imread(dpath0, cv2.IMREAD_UNCHANGED)
        if depth_prev is None:
            print("❌ No se pudo leer depth inicial")
            sys.exit(1)
        depth_prev = depth_prev.astype(np.float32)

    # Loop principal
    max_frames = args.max_frames if args.max_frames is not None else 10**9
    t0 = time.time()

    while idx < max_frames:
        ok, frame = read_frame(left_handle, left_kind, idx)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ts = time.time() - t0

        if method == VOMethod.MONO_2D2D:
            okstep = vo.step_mono(prev_gray, gray, ts)
        else:
            # actualizar depth_prev para cada paso
            if method == VOMethod.STEREO_3D2D:
                # disparidad para el "prev" -> usar right en (idx-1)
                okr, frame_r_prev = read_frame(right_handle, right_kind, idx - 1)
                if not okr:
                    break
                left_prev = prev_gray if prev_gray.ndim == 2 else cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
                right_prev = cv2.cvtColor(frame_r_prev, cv2.COLOR_BGR2GRAY)
                disp = vo.sgbm.compute(left_prev, right_prev).astype(np.float32) / 16.0
                depth_prev = vo._depth_from_disparity(disp)
            else:  # DEPTH
                dpath = args.depth % (idx - 1) if "%" in args.depth else args.depth
                depth_prev = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
                if depth_prev is None:
                    break
                depth_prev = depth_prev.astype(np.float32)

            okstep = vo.step_pnp(prev_gray, gray, depth_prev, ts)

        # si falló el paso, seguir intentando con frames siguientes
        if args.show:
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_gray = gray
        idx += 1

    if args.show:
        cv2.destroyAllWindows()

    # Guardado final (una sola vez)
    traj = np.vstack(vo.traj).astype(np.float32)
    np.save(args.traj_npy, traj)
    print(f"✅ Guardado {args.traj_npy} con {len(traj)} poses")


if __name__ == "__main__":
    main()
