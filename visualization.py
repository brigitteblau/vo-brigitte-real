# visualization.py
import numpy as np

# Intentamos Pangolin; si no está, caemos a Matplotlib (Windows-friendly)
try:
    import pypangolin as pango
    from OpenGL.GL import (
        glEnable, glClear, glClearColor, glBegin, glEnd, glVertex3d, glLineWidth,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LINES, GL_DEPTH_TEST
    )
    _HAS_PANGO = True
except Exception:
    _HAS_PANGO = False
    import matplotlib
    matplotlib.use("TkAgg", force=True)  # asegura ventana en Windows
    import matplotlib.pyplot as plt

# Estado global
_STATE = {
    "pts": [],         # lista de [x,y,z]
    "fig": None,
    "ax": None,
    "pango_ready": False,
    "pango_cam": None,
    "pango_disp": None,
    "title": "Trayectoria",
}

def _xyz_from_pose(p):
    """Acepta [x,y,z] o matriz 4x4 y devuelve np.array([x,y,z])."""
    if p is None:
        return None
    a = np.asarray(p)
    if a.ndim == 1 and a.size >= 3:
        return a[:3].astype(float)
    if a.shape == (4, 4):
        return a[:3, 3].astype(float)
    raise ValueError("Pose no reconocida. Usa [x,y,z] o matriz 4x4.")

# -------- Pangolin (si existe) --------
def _init_pangolin(width=1280, height=720, title="Trajectory 3D"):
    pango.CreateWindowAndBind(title, width, height)
    glEnable(GL_DEPTH_TEST)
    pm = pango.ProjectionMatrix(width, height, 420, 420, width/2, height/2, 0.1, 1000)
    mv = pango.ModelViewLookAt(0.0, 3.0, 6.0, 0.0, 0.0, 0.0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)
    handler = pango.Handler3D(s_cam)
    d_cam = (pango.CreateDisplay()
             .SetBounds(pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach(1), -width/height)
             .SetHandler(handler))
    _STATE["pango_cam"] = s_cam
    _STATE["pango_disp"] = d_cam
    _STATE["pango_ready"] = True

def _pangolin_draw():
    pts = np.asarray(_STATE["pts"], dtype=float)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    _STATE["pango_disp"].Activate(_STATE["pango_cam"])
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glLineWidth(2)
    if len(pts) >= 2:
        glBegin(GL_LINES)
        for i in range(len(pts) - 1):
            x1, y1, z1 = pts[i]
            x2, y2, z2 = pts[i+1]
            glVertex3d(x1, y1, z1)
            glVertex3d(x2, y2, z2)
        glEnd()
    pango.glDrawAxis(1.0)
    pango.FinishFrame()

# -------- Matplotlib (fallback 2D) --------
def _init_matplotlib(title="Trayectoria (X-Z)"):
    plt.ion()
    fig = plt.figure(title)
    ax = fig.add_subplot(111)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True)
    _STATE["fig"] = fig
    _STATE["ax"] = ax
    _STATE["title"] = title
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

def _matplotlib_draw():
    pts = np.asarray(_STATE["pts"], dtype=float)
    if pts.size == 0:
        return
    ax = _STATE["ax"]; fig = _STATE["fig"]
    ax.cla()
    ax.set_title(_STATE["title"])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.grid(True)
    ax.plot(pts[:, 0], pts[:, 2], "-")  # X vs Z
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

# -------- API pública --------
def init_traj_view(title="Trayectoria"):
    if _HAS_PANGO:
        _init_pangolin(title=title)
    else:
        _init_matplotlib(title if title else "Trayectoria (X-Z)")

def traj_update_from_pose(pose_or_T):
    xyz = _xyz_from_pose(pose_or_T)
    if xyz is None:
        return
    _STATE["pts"].append(xyz)
    if _HAS_PANGO:
        if not _STATE["pango_ready"]:
            _init_pangolin(title="Trajectory 3D")
        _pangolin_draw()
    else:
        if _STATE["fig"] is None or _STATE["ax"] is None:
            _init_matplotlib("Trayectoria (X-Z)")
        _matplotlib_draw()

def save_trajectory_npy(path="trajectory.npy"):
    arr = np.asarray(_STATE["pts"], dtype=float)
    np.save(path, arr)
    print(f"✅ Guardado {path} con {len(arr)} poses")
