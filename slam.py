# slam.py
import numpy as np
import cv2
import copy
from typing import List
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from vo_hibrido import HybridVOcd as HybridVO         
from optimizer import BundleAdjustment
from utils import project_points, angle_between
from eval.associate import read_file_list, associate


"""
Heavily inspired by the ORB-SLAM approach.
"""


class Frame:  # keyframe is the same
    def __init__(self, timestamp, pose, kpts, descs, points_3d, img=None):
        self.timestamp = timestamp
        self.pose = pose
        # (N,2) storing only the position because cv2.Keypoint is not pickleable directly
        self.kpts = kpts
        self.descs = descs               # (N, 32)
        self.points_3d = points_3d       # 3D coordinates in world frame
        self.img = img                   # store image for debugging purposes

        assert len(kpts) == len(descs) == len(points_3d)
        # index of associated map point for each keypoint. None => no association
        self.map_point_ids = [None] * len(kpts)


class Tracking:
    def __init__(
        self,
        cx,
        cy,
        fx,
        baseline=1,
        initial_pose=np.eye(4),
        visualize=True,
        save_path=True,
    ) -> None:
        # --------- Camera Parameters ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.baseline = baseline

        # Construimos la intrínseca ANTES de crear el VO y se la pasamos
        self.K = np.array(
            [[fx, 0, cx],
             [0, fx, cy],
             [0,  0,  1]],
            dtype=float
        )

        # --------- Visualization / logging ---------
        self.visualize = visualize
        self.save_path = save_path
        if self.save_path:
            with open("poses.txt", "w") as f:
                f.write("# timestamp x y z qx qy qz qw\n")

        # --------- VO helpers ---------
        self.vo = HybridVO(self.K)  # contains various helper functions

        # --------- Multiprocessing to communicate with Mapping thread ---------
        self.new_keyframe_event = False
        self.new_keyframe = None
        self.processed_new_keyframe = None

        # By default, mapping is done in a separate process (see main_stereo_slam.py)
        self.disable_multiprocessing = True
        if self.disable_multiprocessing:
            self.map = Map(cx, cy, fx)  # run mapping in the same thread

        # --------- State ---------
        self.frames: List[Frame] = []
        self.keyframes: List[Frame] = []
        self.map_points: List[MapPoint] = []
        self.map_point_descs = np.empty((0, 32), dtype=np.uint8)  # ORB descriptors

        self.frames_elapsed_since_keyframe = 0
        self.initial_pose = initial_pose
        self.tracking_success = True

        self.img_width = None
        self.img_height = None

    def predict_pose(self, pose_t_2, pose_t_1):
        """
        Constant velocity model (simple): usa la última pose.
        """
        return pose_t_1

    def synchronize(self, keyframes, map_points):
        """
        Synchronize the keyframes and map points. Updates the frame poses.
        In multiprocessing, the higher-level thread should call this function
        after the mapping thread has optimized the map points.
        """
        self.map_points = copy.deepcopy(map_points)  # optimized map points
        self.keyframes = copy.deepcopy(keyframes)

        kf_id = 0
        # Update frame poses
        for i, frame in enumerate(self.frames):
            # find nearest corresponding keyframe
            while kf_id + 1 < len(keyframes) and keyframes[kf_id + 1].timestamp <= frame.timestamp:
                kf_id += 1

            if frame.timestamp == keyframes[kf_id].timestamp:  # frame is a keyframe
                self.frames[i].pose = keyframes[kf_id].pose
            else:
                # correct by the difference in keyframe poses
                # old_T_new = old_T_w @ w_T_new
                old_T_new = np.linalg.inv(self.keyframes[kf_id].pose) @ keyframes[kf_id].pose
                # w_T_new = w_T_old @ old_T_new
                self.frames[i].pose = frame.pose @ old_T_new

    def track(self, img_t, depth_t, timestamp):
        """
        - Solo consideramos features con depth válido (no NaN)
        """
        print(f"Frame {len(self.frames)}")
        self.img_width = img_t.shape[1]
        self.img_height = img_t.shape[0]

        # ---------- Convert Image to Grayscale -----------
        img_gray_t = self.vo._convert_grayscale(img_t)
        kpts_t, desc_t = self.vo._compute_orb(img_gray_t)
        kpts_t = np.array([kpt.pt for kpt in kpts_t])  # (N, 2)
        points_3d_k = self.vo._project_2d_kpts_to_3d(depth_t, kpts_t)  # (N, 3)

        # --------- Eliminate features with invalid depth -----------
        mask = np.isnan(points_3d_k).any(axis=1)
        kpts_t = np.array(kpts_t)[~mask]
        desc_t = desc_t[~mask]
        points_3d_k = points_3d_k[~mask]

        # ---------- Predict Pose using constant velocity model -----------
        if len(self.frames) >= 2:
            predicted_pose = self.predict_pose(self.frames[-2].pose, self.frames[-1].pose)
        else:
            predicted_pose = self.initial_pose

        # --------- Convert 3D points to world frame -----------
        points_3d_w = (
            predicted_pose @ np.hstack([points_3d_k, np.ones((len(points_3d_k), 1))]).T
        ).T[:, :3]

        # ---------- Create frame with all the information -----------
        curr_frame = Frame(timestamp, predicted_pose, kpts_t, desc_t, points_3d_w, img_t)

        # ---------- Initialize Local Map -----------
        if len(self.frames) == 0:
            keyframe = self.add_keyframe_tracking(curr_frame)
            self.send_keyframe_to_mapping(curr_frame)
            self.frames.append(keyframe)
            return

        assert len(self.map.map_points) == len(self.map_points)

        # ---------- Compute Visible Map Points in previous frame from current frame -----------
        prev_frame = self.frames[-1]
        curr_frame = self.match_map_points(prev_frame, curr_frame)

        if not self.tracking_success:
            print("relocalizing")
            self.tracking_success = True

        # ---------- Solve for Motion -----------
        # p_2d: dónde se ven en la imagen los landmarks
        # p_3d_w: coordenadas 3D en el mapa
        p_2d = curr_frame.kpts
        p_3d_w = np.array(
            [
                (self.map_points[map_point_id].position if map_point_id is not None else [0, 0, 0])
                for map_point_id in curr_frame.map_point_ids
            ]
        )

        # ---------- Filter out points that are not map points -----------
        assert len(p_2d) == len(p_3d_w)
        valid_pts_mask = np.array([map_point is not None for map_point in curr_frame.map_point_ids])
        p_2d = p_2d[valid_pts_mask]
        p_3d_w = p_3d_w[valid_pts_mask]
        k_T_w = np.linalg.inv(curr_frame.pose)
        p_3d_k = (k_T_w @ np.hstack([p_3d_w, np.ones((len(p_3d_w), 1))]).T).T[:, :3]

        print("Total map points tracked", len(p_2d))
        # --------- Estimate Camera Pose by minimizing reprojection error -----------
        T = self.vo._minimize_reprojection_error(p_2d, p_3d_k)
        estimated_pose = prev_frame.pose @ T
        curr_frame.pose = estimated_pose

        # ---------- Determine if is keyframe -----------
        if len(p_2d) < 150 or self.frames_elapsed_since_keyframe > 20:
            keyframe = self.add_keyframe_tracking(curr_frame)
            self.send_keyframe_to_mapping(curr_frame)  # frame without updated map points
            curr_frame = keyframe

        # ---------- Add frame to list of frames-----------
        self.frames.append(curr_frame)
        self.frames_elapsed_since_keyframe += 1

        if self.save_path:
            with open("poses.txt", "a") as f:
                pose = self.frames[-1].pose
                position = pose[0:3, 3]
                quaternion = Rotation.from_matrix(pose[0:3, 0:3]).as_quat()
                pose_list = list(position) + list(quaternion)
                f.write(f"{timestamp} {' '.join(map(str, pose_list))}\n")

    def send_keyframe_to_mapping(self, keyframe):
        print("adding new keyframe")
        if self.disable_multiprocessing:
            self.map.add_keyframe_mapping(keyframe)
            self.map.optimize()
            self.synchronize(self.map.keyframes, self.map.map_points)
            print("map done optimizing")
            import time
            time.sleep(1)
        else:
            self.new_keyframe_event = True
            self.new_keyframe = keyframe  # needs to be grabbed by mapping thread

    def add_keyframe_tracking(self, curr_frame):
        """
        Code identical to add_keyframe_mapping, but function exists on the tracking side to reduce
        synchronization.
        """
        self.frames_elapsed_since_keyframe = 0

        # Use a temporary frame, since the original new frame must be untouched when sent to mapping
        frame = copy.deepcopy(curr_frame)
        tmp_keyframes = self.keyframes.copy()
        tmp_keyframes.append(frame)

        for i, map_point_id in enumerate(frame.map_point_ids):
            if map_point_id is None:
                frame.map_point_ids[i] = len(self.map_points)
                self.map_points.append(
                    MapPoint(
                        frame.points_3d[i], frame.descs[i], len(self.keyframes), frame.pose[:3, 3]
                    )
                )
                self.map_point_descs = np.vstack(
                    (self.map_point_descs, frame.descs[i].reshape(1, -1)), dtype=np.uint8
                )
            else:
                # ------ Update mean viewing direction for map Point ---------
                self.map_points[map_point_id].add_viewed_keyframe(
                    len(self.keyframes), tmp_keyframes
                )

        self.keyframes.append(frame)
        return frame

    def match_map_points(self, prev_frame, curr_frame):
        """
        Fills curr_frame with corresponding map points ids from prev_frame.
        Exploita la geometría: proyecta puntos y filtra por FoV y ángulo de visión.
        """
        # Project the valid map points
        valid_mask = np.array([map_point is not None for map_point in prev_frame.map_point_ids])

        map_points = np.array(
            [
                (
                    self.map_points[map_point_id].position
                    if map_point_id is not None
                    else np.zeros(3)
                )
                for map_point_id in prev_frame.map_point_ids
            ]
        )

        # curr_frame.pose is the predicted pose of the current frame
        projected_points = project_points(map_points, curr_frame.pose, self.K)

        # Mask out the points that are out of image bounds
        in_bounds_mask = (
            (0 <= projected_points[:, 0])
            & (projected_points[:, 0] < self.img_width)
            & (0 <= projected_points[:, 1])
            & (projected_points[:, 1] < self.img_height)
        )

        # Filter by mean viewing angle
        mean_viewing_angle = np.array(
            [
                (
                    angle_between(
                        self.map_points[map_point_id].mean_viewing_dir,
                        self.map_points[map_point_id].position - curr_frame.pose[:3, 3],
                    )
                    if map_point_id is not None
                    else 0
                )
                for map_point_id in prev_frame.map_point_ids
            ]
        )
        mean_viewing_angle_mask = mean_viewing_angle < np.pi / 3

        final_mask = valid_mask & in_bounds_mask & mean_viewing_angle_mask

        matched_points, good_matches = self.get_matched_points(final_mask, prev_frame, curr_frame)

        # ---------- Add Map Points to Frame (curr_frame) -----------
        map_point_ids_set = set(curr_frame.map_point_ids)
        for idx, map_point_id in matched_points.items():
            if curr_frame.map_point_ids[idx] is None and map_point_id not in map_point_ids_set:
                curr_frame.map_point_ids[idx] = map_point_id
                map_point_ids_set.add(map_point_id)

        # Si hay pocos matches, relajamos criterios
        if len(matched_points) < 15:
            final_mask = valid_mask
            matched_points, good_matches = self.get_matched_points(final_mask, prev_frame, curr_frame)
            print("not enough corresponding points found, expanding search area and tracked",
                  len(matched_points))
            if len(matched_points) < 15:
                self.tracking_success = False

        if self.visualize:
            prev_frame_cv_kpts = [
                cv2.KeyPoint(kpt[0], kpt[1], 1) for kpt in prev_frame.kpts[final_mask]
            ]
            curr_frame_cv_kpts = [cv2.KeyPoint(kpt[0], kpt[1], 1) for kpt in curr_frame.kpts]
            output_image = cv2.drawMatches(
                prev_frame.img,
                prev_frame_cv_kpts,
                curr_frame.img,
                curr_frame_cv_kpts,
                good_matches,
                None,
            )
            cv2.imwrite(f"orb_tracking/{len(self.frames)}.png", output_image)
            cv2.imshow("Tracked ORB", output_image)
            cv2.waitKey(1)

        # asegurar que se reflejen los matches
        for idx, map_point_id in matched_points.items():
            if curr_frame.map_point_ids[idx] is None and map_point_id not in map_point_ids_set:
                curr_frame.map_point_ids[idx] = map_point_id
                map_point_ids_set.add(map_point_id)

        return curr_frame

    def get_matched_points(self, mask, prev_frame, curr_frame):
        prev_kpts = prev_frame.kpts[mask]
        prev_descs = prev_frame.descs[mask]
        prev_map_point_ids = np.array(prev_frame.map_point_ids)[mask].astype(int)

        good_matches = self.vo._get_matches(
            prev_kpts, curr_frame.kpts, prev_descs, curr_frame.descs
        )
        print("good matches", len(good_matches))

        matched_points = {}
        for m in good_matches:
            matched_points[m.trainIdx] = prev_map_point_ids[m.queryIdx]

        return matched_points, good_matches


class MapPoint:
    def __init__(self, position, desc, keyframe_id, keyframe_position):
        self.position = position  # world coordinate system
        self.desc = desc
        self.mean_viewing_dir = position - keyframe_position  # Mean viewing direction
        n = np.linalg.norm(self.mean_viewing_dir)
        if n > 1e-12:
            self.mean_viewing_dir /= n
        self.observed_keyframe_ids = [keyframe_id]  # list of keyframe ids that observed this map point

    def add_viewed_keyframe(self, keyframe_id, keyframes):
        """
        Adding the keyframe computes the mean viewing direction, and selects the best keyframe
        """
        self.observed_keyframe_ids.append(keyframe_id)
        if len(self.observed_keyframe_ids) != len(set(self.observed_keyframe_ids)):
            print("what have you done...")

        # ------ Compute Mean Viewing Direction ---------
        mean_dir = np.array([0.0, 0.0, 0.0])
        for kfid in self.observed_keyframe_ids:
            keyframe = keyframes[kfid]
            v = self.position - keyframe.pose[:3, 3]
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                v = v / nv
                mean_dir += v
        n = np.linalg.norm(mean_dir)
        if n > 1e-12:
            mean_dir /= n
        self.mean_viewing_dir = mean_dir


class Map:
    def __init__(self, cx, cy, fx):
        self.keyframes: List[Frame] = []
        self.map_points: List[MapPoint] = []
        self.ba = BundleAdjustment(cx, cy, fx)

    def add_keyframe_mapping(self, frame: Frame) -> None:
        """
        For the map points that are None, add these
        """
        tmp_keyframes = self.keyframes.copy()
        tmp_keyframes.append(frame)
        for i, map_point_id in enumerate(frame.map_point_ids):
            if map_point_id is None:
                frame.map_point_ids[i] = len(self.map_points)
                self.map_points.append(
                    MapPoint(
                        frame.points_3d[i], frame.descs[i], len(self.keyframes), frame.pose[:3, 3]
                    )
                )
            else:
                # ------ Update mean viewing direction for MapPoint ---------
                if (self.map_points[map_point_id].position - frame.points_3d[i] > 0.1).any():
                    print(
                        f"LARGE difference in landmark position "
                        f"{self.map_points[map_point_id].position - frame.points_3d[i]}"
                    )
                    continue
                self.map_points[map_point_id].add_viewed_keyframe(
                    len(self.keyframes), tmp_keyframes
                )
        self.keyframes.append(frame)

    def optimize(self):
        if len(self.keyframes) < 5:
            return
        self.ba.optimize(self.keyframes, self.map_points)  # inplace update


if __name__ == "__main__":
    # --------- Load Dataset Images ---------
    # Taken from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y (no lo usamos, K usa fx para x e y)
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5000  # for the 16-bit PNG files

    # dataset_name = "rgbd_dataset_freiburg1_xyz"   # para debug rápido
    # dataset_name = "rgbd_dataset_freiburg1_rpy"
    dataset_name = "rgbd_dataset_freiburg1_desk"

    depth_image_paths = read_file_list(f"{dataset_name}/depth.txt")
    rgb_image_paths = read_file_list(f"{dataset_name}/rgb.txt")
    gt_paths = read_file_list(f"{dataset_name}/groundtruth.txt")

    print("Associating depth and rgb images")
    matches = associate(depth_image_paths, rgb_image_paths, 0.0, 0.02)
    gt_matches = associate(depth_image_paths, gt_paths, 0.0, 0.02)

    depth_images = []
    rgb_images = []
    timestamps = []
    gt_poses = []

    gt_dict = {}
    with open(f"{dataset_name}/groundtruth.txt") as file:
        for line in file:
            if line.startswith("#"):
                continue
            pose = np.eye(4)
            pose_list = list(map(float, line.split()))
            pose[:3, 3] = pose_list[1:4]
            pose[:3, :3] = R.from_quat(pose_list[4:]).as_matrix()
            gt_dict[float(pose_list[0])] = pose

    for depth_timestamp, rgb_timestamp in matches.items():
        if depth_timestamp not in gt_matches:
            continue

        depth_image = (
            cv2.imread(
                dataset_name + "/" + depth_image_paths[depth_timestamp][0],
                cv2.IMREAD_UNCHANGED,
            )
            / factor
        )
        # A pixel value of 0 means missing value/no data.
        depth_image[depth_image == 0] = np.nan   # NumPy 2.x

        rgb_image = cv2.imread(dataset_name + "/" + rgb_image_paths[rgb_timestamp][0])

        depth_images.append(depth_image)
        rgb_images.append(rgb_image)
        timestamps.append(depth_timestamp)
        gt_poses.append(gt_dict[gt_matches[depth_timestamp]])

    initial_pose = gt_poses[0]
    tracker = Tracking(cx, cy, fx, 1, initial_pose)

    # Si querés limitar frames para test rápido, descomentá:
    # max_frames = 30
    # for i, (rgb_img, depth_img, timestamp) in enumerate(zip(rgb_images, depth_images, timestamps)):
    #     if i >= max_frames: break
    #     tracker.track(rgb_img, depth_img, timestamp)

    for rgb_img, depth_img, timestamp in zip(rgb_images, depth_images, timestamps):
        tracker.track(rgb_img, depth_img, timestamp)
