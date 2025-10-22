"""
Bundle Adjustment optimizer con soporte híbrido:
- Usa g2o si está disponible (óptimo)
- Fallback automático a scipy (compatible Windows)
"""
import numpy as np

# Detectar backend disponible
try:
    import g2o
    BACKEND = "g2o"
    print("Using g2o backend for Bundle Adjustment")
except ImportError:
    BACKEND = "scipy"
    print("g2o not found, using scipy backend for Bundle Adjustment")
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

np.random.seed(0)


class BundleAdjustment:
    def __init__(self, cx, cy, fx) -> None:
        """
        Inicializa el optimizador de Bundle Adjustment
        
        Args:
            cx: Centro óptico en x
            cy: Centro óptico en y  
            fx: Distancia focal (asumimos fx = fy)
        """
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.verbose = True
        self.backend = BACKEND

    def optimize(self, keyframes, map_points, num_iterations=30, inplace=True):
        """
        Optimiza poses de keyframes y posiciones de puntos 3D
        
        Args:
            keyframes: Lista de frames (con pose y map_point_ids)
            map_points: Lista de MapPoint (con position)
            num_iterations: Número máximo de iteraciones
            inplace: Si True, modifica keyframes/map_points directamente
        
        Returns:
            keyframes, map_points optimizados
        """
        if self.backend == "g2o":
            return self._optimize_g2o(keyframes, map_points, num_iterations, inplace)
        else:
            return self._optimize_scipy(keyframes, map_points, num_iterations, inplace)

    # ==================== G2O BACKEND ====================
    
    def _optimize_g2o(self, keyframes, map_points, num_iterations, inplace):
        """Optimización usando g2o (método original)"""
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Add Camera
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # --------- Add Camera Poses ---------
        for i, keyframe in enumerate(keyframes):
            curr_pose = keyframe.pose
            curr_pose = np.linalg.inv(curr_pose)
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(g2o.SE3Quat(curr_pose[:3, :3], curr_pose[:3, 3]))
            if i == 0:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

        # --------- Add 3D Landmark position ---------
        is_valid = [True] * len(map_points)
        for i, map_point in enumerate(map_points):
            if len(map_point.observed_keyframe_ids) < 3:  # ignore unreliable points
                is_valid[i] = False
                continue
            
            point_3d = map_point.position
            vp = g2o.VertexPointXYZ()
            vp.set_id(len(keyframes) + i)
            vp.set_marginalized(True)
            vp.set_estimate(point_3d)
            optimizer.add_vertex(vp)

        # --------- Add Edges (observations) ---------
        for i, keyframe in enumerate(keyframes):
            for j, map_point_id in enumerate(keyframe.map_point_ids):
                if map_point_id is None or not is_valid[map_point_id]:
                    continue
                
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, optimizer.vertex(len(keyframes) + map_point_id))
                edge.set_vertex(1, optimizer.vertex(i))
                edge.set_measurement(keyframe.kpts[j])
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

        print(f"[g2o] num vertices: {len(optimizer.vertices())}, num edges: {len(optimizer.edges())}")
        optimizer.initialize_optimization()
        optimizer.optimize(num_iterations)

        # ----------- Update State -----------
        print("Done optimization (g2o)")
        for i in range(len(keyframes)):
            if self.verbose:
                print(f"KF {i} - Before BA: {keyframes[i].pose[:3, 3]}")
                new_pose = np.linalg.inv(optimizer.vertex(i).estimate().to_homogeneous_matrix())
                print(f"KF {i} - After BA:  {new_pose[:3, 3]}")
            keyframes[i].pose = np.linalg.inv(
                optimizer.vertex(i).estimate().to_homogeneous_matrix()
            )

        for i in range(len(map_points)):
            if is_valid[i]:
                map_points[i].position = optimizer.vertex(len(keyframes) + i).estimate()

        return keyframes, map_points

    # ==================== SCIPY BACKEND ====================
    
    def _optimize_scipy(self, keyframes, map_points, num_iterations, inplace):
        """Optimización usando scipy (fallback compatible con Windows)"""
        if len(keyframes) < 2 or len(map_points) < 10:
            print("Not enough keyframes or map points for BA")
            return keyframes, map_points
        
        # Filtrar puntos válidos (mismo criterio que g2o)
        is_valid = [len(mp.observed_keyframe_ids) >= 3 for mp in map_points]
        valid_point_count = sum(is_valid)
        
        if valid_point_count < 5:
            print(f"Not enough valid map points for BA ({valid_point_count})")
            return keyframes, map_points
        
        print(f"[scipy] Bundle Adjustment: {len(keyframes)} keyframes, "
              f"{valid_point_count}/{len(map_points)} valid map points")
        
        # Preparar datos
        camera_params, points_3d, camera_indices, point_indices, points_2d, valid_map = \
            self._prepare_bundle_adjustment_data(keyframes, map_points, is_valid)
        
        if len(points_2d) < 20:
            print(f"Not enough observations for BA ({len(points_2d)})")
            return keyframes, map_points
        
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]
        
        # Parámetros iniciales: [camera_params (excepto primera), points_3d]
        # Primera cámara está fija
        x0 = np.hstack((camera_params[1:].ravel(), points_3d.ravel()))
        
        # Sparsity pattern
        A = self._bundle_adjustment_sparsity(
            n_cameras, n_points, camera_indices, point_indices
        )
        
        print(f"[scipy] Optimizing {len(x0)} parameters with {len(points_2d)} observations")
        
        # Optimización con Huber loss (robusto a outliers, como g2o)
        res = least_squares(
            self._residuals_scipy,
            x0,
            jac_sparsity=A,
            verbose=0,
            loss='huber',  # equivalente a RobustKernelHuber de g2o
            f_scale=1.0,   # threshold del huber kernel
            x_scale='jac',
            ftol=1e-4,
            method='trf',
            max_nfev=num_iterations * 10,  # iterations × puntos por iteración
            args=(camera_params[0], n_cameras, n_points, 
                  camera_indices, point_indices, points_2d)
        )
        
        # Actualizar poses y puntos
        self._update_from_optimization_scipy(
            res.x, camera_params[0], n_cameras, n_points, 
            keyframes, map_points, valid_map
        )
        
        print(f"Done optimization (scipy): cost {res.cost:.4f}, success: {res.success}")
        
        return keyframes, map_points
    
    def _prepare_bundle_adjustment_data(self, keyframes, map_points, is_valid):
        """
        Prepara los datos para scipy
        
        Returns:
            camera_params: (n_cameras, 6) - [rx, ry, rz, tx, ty, tz]
            points_3d: (n_valid_points, 3) - Solo puntos válidos
            camera_indices: (n_observations,)
            point_indices: (n_observations,) - Índices en el array de puntos válidos
            points_2d: (n_observations, 2)
            valid_map: dict - Mapeo de map_point_id original a índice en points_3d
        """
        n_cameras = len(keyframes)
        
        # Convertir poses a axis-angle
        camera_params = np.zeros((n_cameras, 6))
        for i, kf in enumerate(keyframes):
            # g2o usa pose inversa (world to camera), scipy usa camera to world
            # Mantener consistencia con g2o
            pose_inv = np.linalg.inv(kf.pose)
            R = pose_inv[:3, :3]
            t = pose_inv[:3, 3]
            
            rvec = self._rotation_matrix_to_axis_angle(R)
            camera_params[i] = np.hstack([rvec, t])
        
        # Crear mapeo de puntos válidos
        valid_map = {}
        valid_idx = 0
        points_3d_list = []
        
        for i, (mp, valid) in enumerate(zip(map_points, is_valid)):
            if valid:
                valid_map[i] = valid_idx
                points_3d_list.append(mp.position)
                valid_idx += 1
        
        points_3d = np.array(points_3d_list)
        
        # Recolectar observaciones
        camera_indices = []
        point_indices = []
        points_2d = []
        
        for cam_idx, kf in enumerate(keyframes):
            for kpt_idx, map_point_id in enumerate(kf.map_point_ids):
                if map_point_id is not None and map_point_id in valid_map:
                    camera_indices.append(cam_idx)
                    point_indices.append(valid_map[map_point_id])
                    points_2d.append(kf.kpts[kpt_idx])
        
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_2d = np.array(points_2d)
        
        return camera_params, points_3d, camera_indices, point_indices, points_2d, valid_map
    
    def _residuals_scipy(self, params, fixed_cam, n_cameras, n_points, 
                        camera_indices, point_indices, points_2d):
        """Calcula residuos de reproyección (compatible con primera cámara fija)"""
        # Reconstruir todos los parámetros de cámara
        camera_params = np.zeros((n_cameras, 6))
        camera_params[0] = fixed_cam  # primera cámara fija
        camera_params[1:] = params[:(n_cameras-1) * 6].reshape((n_cameras-1, 6))
        
        # Puntos 3D
        points_3d = params[(n_cameras-1) * 6:].reshape((n_points, 3))
        
        # Proyectar
        points_proj = self._project_scipy(
            points_3d[point_indices], 
            camera_params[camera_indices]
        )
        
        # Residuos
        residuals = (points_proj - points_2d).ravel()
        
        return residuals
    
    def _bundle_adjustment_sparsity(self, n_cameras, n_points, 
                                   camera_indices, point_indices):
        """Matriz de sparsity del Jacobiano (sin primera cámara)"""
        m = camera_indices.size * 2
        n = (n_cameras - 1) * 6 + n_points * 3  # -1 porque primera cámara fija
        A = lil_matrix((m, n), dtype=int)
        
        i = np.arange(camera_indices.size)
        
        # Derivadas respecto a cámaras (excepto la primera)
        for obs_idx in range(len(camera_indices)):
            cam_idx = camera_indices[obs_idx]
            if cam_idx > 0:  # Skip primera cámara
                cam_param_start = (cam_idx - 1) * 6
                for s in range(6):
                    A[2 * obs_idx, cam_param_start + s] = 1
                    A[2 * obs_idx + 1, cam_param_start + s] = 1
        
        # Derivadas respecto a puntos 3D
        for s in range(3):
            A[2 * i, (n_cameras - 1) * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, (n_cameras - 1) * 6 + point_indices * 3 + s] = 1
        
        return A
    
    def _update_from_optimization_scipy(self, params, fixed_cam, n_cameras, n_points,
                                       keyframes, map_points, valid_map):
        """Actualiza keyframes y map_points con resultados de scipy"""
        # Reconstruir parámetros de cámara
        camera_params = np.zeros((n_cameras, 6))
        camera_params[0] = fixed_cam
        camera_params[1:] = params[:(n_cameras-1) * 6].reshape((n_cameras-1, 6))
        
        points_3d = params[(n_cameras-1) * 6:].reshape((n_points, 3))
        
        # Actualizar keyframes
        for i, kf in enumerate(keyframes):
            rvec = camera_params[i, :3]
            tvec = camera_params[i, 3:]
            
            R = self._axis_angle_to_rotation_matrix(rvec)
            
            # Convertir de vuelta a world-to-camera (pose inversa)
            pose_inv = np.eye(4)
            pose_inv[:3, :3] = R
            pose_inv[:3, 3] = tvec
            
            kf.pose = np.linalg.inv(pose_inv)
            
            if self.verbose and i < 3:
                print(f"KF {i} - After BA (scipy): {kf.pose[:3, 3]}")
        
        # Actualizar map points (solo los válidos)
        for original_id, valid_idx in valid_map.items():
            map_points[original_id].position = points_3d[valid_idx]
    
    def _project_scipy(self, points_3d, camera_params):
        """Proyecta puntos 3D a 2D (batch processing)"""
        n = points_3d.shape[0]
        points_2d = np.zeros((n, 2))
        
        for i in range(n):
            rvec = camera_params[i, :3]
            tvec = camera_params[i, 3:]
            
            # FIX: Asegurarse que rvec es 1D
            rvec = np.atleast_1d(rvec).flatten()
            tvec = np.atleast_1d(tvec).flatten()
            
            # Transformar a coordenadas de cámara
            R = self._axis_angle_to_rotation_matrix(rvec)
            
            # FIX: Asegurarse que points_3d[i] es 1D
            point_3d = np.atleast_1d(points_3d[i]).flatten()
            
            # Verificar que R es 2D y point_3d es 1D
            if R.ndim != 2 or point_3d.ndim != 1:
                print(f"ERROR: R.shape={R.shape}, point_3d.shape={point_3d.shape}")
                points_2d[i] = [1e6, 1e6]
                continue
            
            point_cam = R @ point_3d + tvec
            
            # Proyección perspectiva
            if point_cam[2] > 0.01:
                x = point_cam[0] / point_cam[2]
                y = point_cam[1] / point_cam[2]
                
                points_2d[i, 0] = self.fx * x + self.cx
                points_2d[i, 1] = self.fx * y + self.cy
            else:
                # Punto detrás de la cámara
                points_2d[i] = [1e6, 1e6]
        
        return points_2d
    # ==================== UTILIDADES ====================
    
    @staticmethod
    def _rotation_matrix_to_axis_angle(R):
        """Rodrigues: matriz → axis-angle"""
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        if theta < 1e-6:
            return np.zeros(3)
        
        w = (1 / (2 * np.sin(theta))) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        
        return theta * w
    
    @staticmethod
    def _axis_angle_to_rotation_matrix(rvec):
        """Rodrigues: axis-angle → matriz"""
        theta = np.linalg.norm(rvec)
        
        if theta < 1e-6:
            return np.eye(3)
        
        w = rvec / theta
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return 