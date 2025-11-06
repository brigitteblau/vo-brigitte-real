# optimizer.py
import numpy as np
import g2o

class BundleAdjustment:
    def __init__(self, fx, cx, cy, verbose=False):
        self.fx = float(fx)
        self.cx = float(cx)
        self.cy = float(cy)
        self.verbose = verbose

    def _make_optimizer(self):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        opt.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
        return opt

    def optimize(self, keyframes, map_points, num_iterations=20, sigma_px=1.5, inplace=True):
        """
        keyframes: lista de Keyframe
            - keyframe.pose: (4x4) homogénea. **Convención**: T_cw (cámara←mundo) o T_wc (mundo←cámara).
              Aquí asumimos que VIENE como T_cw (común en VO) y la invertimos para g2o si es necesario.
            - keyframe.kpts: lista de (u, v) floats por punto observado
            - keyframe.map_point_ids: lista paralela con ids de map_points (o None si no hay punto)
        map_points: lista de MapPoint
            - map_point.position: (3,) coords en mundo (np.float64)
            - map_point.observed_keyframe_ids: lista de ids de keyframes donde fue visto (opcional)
        """
        optimizer = self._make_optimizer()

        # --- Camera parameters ---
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0.0)  # baseline=0 (mono)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # --- Add pose vertices ---
        n_poses = len(keyframes)
        for i, kf in enumerate(keyframes):
            # Si tus kf.pose es T_cw (cámara←mundo), g2o espera SE3Quat(mundo→cámara)
            # entonces usamos directamente T_cw para SE3Quat(R, t).
            T_cw = kf.pose.astype(np.float64)

            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            R = T_cw[:3, :3]
            t = T_cw[:3, 3]
            v_se3.set_estimate(g2o.SE3Quat(R, t))
            if i == 0:
                v_se3.set_fixed(True)  # fija el gauge
            optimizer.add_vertex(v_se3)

        # --- Add 3D point vertices ---
        # Requisitos: punto visto en ≥ 2-3 frames (evitá puntos flojos)
        is_valid = [True] * len(map_points)
        point_offset = n_poses

        for pid, mp in enumerate(map_points):
            if getattr(mp, "observed_keyframe_ids", None) and len(mp.observed_keyframe_ids) < 2:
                is_valid[pid] = False
                continue
            if mp.position is None or np.isnan(mp.position).any():
                is_valid[pid] = False
                continue

            vp = g2o.VertexPointXYZ()
            vp.set_id(point_offset + pid)
            vp.set_marginalized(True)
            vp.set_estimate(mp.position.astype(np.float64))
            optimizer.add_vertex(vp)

        # --- Add projection edges (observations) ---
        info = (1.0 / (sigma_px ** 2)) * np.eye(2, dtype=np.float64)

        for i, kf in enumerate(keyframes):
            for j, mp_id in enumerate(kf.map_point_ids):
                if mp_id is None or mp_id < 0:
                    continue
                if mp_id >= len(map_points) or not is_valid[mp_id]:
                    continue
                uv = kf.kpts[j]
                if uv is None:
                    continue
                # uv debe ser (u, v) float
                u, v = float(uv[0]), float(uv[1])

                e = g2o.EdgeProjectXYZ2UV()
                # punto 3D
                e.set_vertex(0, optimizer.vertex(point_offset + mp_id))
                # pose
                e.set_vertex(1, optimizer.vertex(i))
                e.set_measurement(np.array([u, v], dtype=np.float64))
                e.set_information(info)
                e.set_parameter_id(0, 0)  # link cámara

                rk = g2o.RobustKernelHuber()
                rk.set_delta(2.0)  # ~2 px
                e.set_robust_kernel(rk)

                optimizer.add_edge(e)

        if self.verbose:
            print("num vertices:", len(optimizer.vertices()))
            print("num edges:", len(optimizer.edges()))

        optimizer.initialize_optimization()
        # optimizer.set_verbose(self.verbose)
        optimizer.optimize(int(num_iterations))

        # --- Update back results ---
        if inplace:
            # Actualiza poses (T_cw)
            for i, kf in enumerate(keyframes):
                est = optimizer.vertex(i).estimate().to_homogeneous_matrix()
                # est es T_cw (mundo→cámara) en esta convención
                kf.pose = est

            # Actualiza puntos
            for pid, mp in enumerate(map_points):
                if is_valid[pid]:
                    mp.position = optimizer.vertex(point_offset + pid).estimate()
            return keyframes, map_points
        else:
            new_kf = []
            for i, kf in enumerate(keyframes):
                kf2 = kf.copy()
                kf2.pose = optimizer.vertex(i).estimate().to_homogeneous_matrix()
                new_kf.append(kf2)
            new_mp = []
            for pid, mp in enumerate(map_points):
                mp2 = mp.copy()
                if is_valid[pid]:
                    mp2.position = optimizer.vertex(point_offset + pid).estimate()
                new_mp.append(mp2)
            return new_kf, new_mp
