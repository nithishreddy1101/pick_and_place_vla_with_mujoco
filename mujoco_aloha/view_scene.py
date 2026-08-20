"""
view_scene.py – Headless Aloha simulation with:
  • Live camera windows  (OpenCV)
  • Joint-state controller
  • Tkinter GUI with sliders to command every joint in real time

Usage:
    python view_scene.py
    Press Q or close any camera window to quit.

Requirements:
    pip install mujoco opencv-python numpy
    (tkinter ships with CPython – no extra install needed)
"""

import os
import time
import threading
import numpy as np
import mujoco
import cv2
import tkinter as tk
from tkinter import ttk

# ── Config ────────────────────────────────────────────────────────────────────
SCENE_DIR             = os.path.dirname(os.path.abspath(__file__))
SCENE_XML             = os.path.join(SCENE_DIR, "scene.xml")

RENDER_W, RENDER_H    = 640, 360
SIM_STEPS_PER_FRAME   = 5
PRINT_STATE_EVERY_SEC = 0.0    # set to 0 to disable

CAMERAS = [
    "overhead_cam",
    "worms_eye_cam",
    "wrist_cam_left",
    "wrist_cam_right",
]
# ─────────────────────────────────────────────────────────────────────────────


class JointStateController:
    """Reads joint states and commands joint positions via data.ctrl."""

    JOINT_ACTUATOR_MAP = [
        ("left/waist",         "left/waist"),
        ("left/shoulder",      "left/shoulder"),
        ("left/elbow",         "left/elbow"),
        ("left/forearm_roll",  "left/forearm_roll"),
        ("left/wrist_angle",   "left/wrist_angle"),
        ("left/wrist_rotate",  "left/wrist_rotate"),
        ("left/left_finger",   "left/gripper"),
        ("right/waist",        "right/waist"),
        ("right/shoulder",     "right/shoulder"),
        ("right/elbow",        "right/elbow"),
        ("right/forearm_roll", "right/forearm_roll"),
        ("right/wrist_angle",  "right/wrist_angle"),
        ("right/wrist_rotate", "right/wrist_rotate"),
        ("right/left_finger",  "right/gripper"),
    ]

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data  = data
        self._lock = threading.Lock()

        self._joints: list[dict] = []
        for jnt_name, act_name in self.JOINT_ACTUATOR_MAP:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if jnt_id < 0:
                print(f"  [JSC] WARNING: joint '{jnt_name}' not found.")
                continue
            lo = float(model.jnt_range[jnt_id, 0])
            hi = float(model.jnt_range[jnt_id, 1])
            self._joints.append({
                "name":      jnt_name,
                "act_id":    act_id,
                "qpos_addr": model.jnt_qposadr[jnt_id],
                "qvel_addr": model.jnt_dofadr[jnt_id],
                "range":     (lo, hi),
            })

        self._targets: dict[str, float] = {
            j["name"]: float(data.qpos[j["qpos_addr"]]) for j in self._joints
        }
        print(f"  [JointStateController] Tracking {len(self._joints)} joints.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def joints(self) -> list[dict]:
        return self._joints

    def set_target_position(self, joint_name: str, value: float) -> None:
        with self._lock:
            if joint_name in self._targets:
                self._targets[joint_name] = float(value)

    def apply_targets(self) -> None:
        with self._lock:
            for j in self._joints:
                if j["act_id"] >= 0:
                    self.data.ctrl[j["act_id"]] = self._targets[j["name"]]

    def get_state(self) -> list[dict]:
        states = []
        for j in self._joints:
            act_id = j["act_id"]
            effort = float(self.data.actuator_force[act_id]) if act_id >= 0 else 0.0
            with self._lock:
                target = self._targets[j["name"]]
            states.append({
                "name":     j["name"],
                "position": float(self.data.qpos[j["qpos_addr"]]),
                "velocity": float(self.data.qvel[j["qvel_addr"]]),
                "effort":   effort,
                "target":   target,
            })
        return states

    def print_state(self) -> None:
        states = self.get_state()
        print(f"\n{'Joint':<25} {'Pos':>10} {'Vel':>10} {'Effort':>10} {'Target':>10}")
        print("─" * 68)
        for s in states:
            print(f"{s['name']:<25} {s['position']:>10.4f} {s['velocity']:>10.4f} "
                  f"{s['effort']:>10.4f} {s['target']:>10.4f}")


# ── Tkinter GUI ───────────────────────────────────────────────────────────────

class JointControlGUI:
    """
    Tkinter window with one slider per joint.
    Runs in its own daemon thread so it never blocks the sim loop.
    """

    BG          = "#1e1e2e"
    FG          = "#cdd6f4"
    ACCENT      = "#89b4fa"
    SLIDER_BG   = "#313244"
    LEFT_COLOR  = "#a6e3a1"   # green tint for left arm
    RIGHT_COLOR = "#f38ba8"   # red tint for right arm
    LABEL_FONT  = ("Segoe UI", 9)
    TITLE_FONT  = ("Segoe UI", 10, "bold")

    def __init__(self, controller: JointStateController) -> None:
        self.controller = controller
        self._running   = True
        self._vars: dict[str, tk.DoubleVar] = {}
        self._val_labels: dict[str, tk.Label] = {}

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.root = tk.Tk()
        self.root.title("Aloha Joint Controller")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._poll_state()
        self.root.mainloop()

    def _build_ui(self) -> None:
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Label(self.root, text="Aloha Joint Controller",
                       bg=self.BG, fg=self.ACCENT,
                       font=("Segoe UI", 13, "bold"), pady=8)
        hdr.pack(fill="x")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        # ── Scrollable canvas ─────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=self.BG)
        canvas_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(canvas_frame, bg=self.BG, highlightthickness=0,
                           width=420, height=600)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=self.BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        # Mouse-wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Build one row per joint ───────────────────────────────────────────
        current_arm = None
        for j in self.controller.joints:
            arm = j["name"].split("/")[0]   # "left" or "right"
            color = self.LEFT_COLOR if arm == "left" else self.RIGHT_COLOR

            # Section header when arm changes
            if arm != current_arm:
                current_arm = arm
                lbl = tk.Label(inner,
                               text=f"{'◀  Left Arm  ▶' if arm == 'left' else '◀  Right Arm  ▶'}",
                               bg=self.BG, fg=color,
                               font=self.TITLE_FONT, pady=6)
                lbl.pack(fill="x", padx=12)

            lo, hi = j["range"]
            cur    = float(self.controller.data.qpos[j["qpos_addr"]])

            row = tk.Frame(inner, bg=self.SLIDER_BG,
                           highlightbackground=color,
                           highlightthickness=1)
            row.pack(fill="x", padx=12, pady=3, ipady=4)

            # Joint label
            jnt_short = j["name"].split("/")[1]   # e.g. "shoulder"
            tk.Label(row, text=jnt_short, bg=self.SLIDER_BG, fg=color,
                     font=self.LABEL_FONT, width=14, anchor="w"
                     ).grid(row=0, column=0, padx=(8, 0))

            # Slider
            var = tk.DoubleVar(value=cur)
            self._vars[j["name"]] = var

            slider = tk.Scale(
                row, variable=var,
                from_=lo, to=hi,
                resolution=0.001,
                orient="horizontal",
                length=220,
                bg=self.SLIDER_BG, fg=self.FG,
                troughcolor=self.BG,
                activebackground=self.ACCENT,
                highlightthickness=0,
                showvalue=False,
                command=lambda val, name=j["name"]: self._on_slider(name, val),
            )
            slider.grid(row=0, column=1, padx=6)

            # Numeric value label
            val_lbl = tk.Label(row, text=f"{cur:.3f}",
                               bg=self.SLIDER_BG, fg=self.FG,
                               font=self.LABEL_FONT, width=7)
            val_lbl.grid(row=0, column=2, padx=(0, 8))
            self._val_labels[j["name"]] = val_lbl

            # Range hint
            tk.Label(row, text=f"[{lo:.2f}, {hi:.2f}]",
                     bg=self.SLIDER_BG, fg="#6c7086",
                     font=("Segoe UI", 7)).grid(row=1, column=1, sticky="w", padx=6)

        # ── Reset button ──────────────────────────────────────────────────────
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=4)
        btn = tk.Button(self.root, text="Reset to Neutral Pose",
                        bg=self.ACCENT, fg=self.BG,
                        font=("Segoe UI", 9, "bold"),
                        relief="flat", padx=10, pady=4,
                        command=self._reset_neutral)
        btn.pack(pady=(0, 10))

    def _on_slider(self, joint_name: str, value: str) -> None:
        v = float(value)
        self.controller.set_target_position(joint_name, v)
        lbl = self._val_labels.get(joint_name)
        if lbl:
            lbl.config(text=f"{v:.3f}")

    def _reset_neutral(self) -> None:
        """Reset all sliders to the neutral-pose keyframe values."""
        # Neutral pose qpos from keyframe_ctrl.xml (14 values, fingers last)
        neutral_qpos = [
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084,   # left arm
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084,   # right arm
        ]
        for j, val in zip(self.controller.joints, neutral_qpos):
            self.controller.set_target_position(j["name"], val)
            var = self._vars.get(j["name"])
            if var:
                var.set(val)
            lbl = self._val_labels.get(j["name"])
            if lbl:
                lbl.config(text=f"{val:.3f}")

    def _poll_state(self) -> None:
        """Periodically sync slider display to actual qpos (feedback)."""
        if not self._running:
            return
        for j in self.controller.joints:
            actual = float(self.controller.data.qpos[j["qpos_addr"]])
            lbl = self._val_labels.get(j["name"])
            # Only update label – don't move slider (that would fight the user)
            if lbl:
                lbl.config(text=f"{actual:.3f}")
        self.root.after(200, self._poll_state)   # refresh every 200 ms

    def _on_close(self) -> None:
        self._running = False
        self.root.destroy()

    def is_running(self) -> bool:
        return self._running


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading model: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    # Apply safe neutral pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"  Applied keyframe: neutral_pose (id={key_id})")
    else:
        print("  WARNING: 'neutral_pose' keyframe not found.")

    # Build controller and GUI
    controller = JointStateController(model, data)
    gui        = JointControlGUI(controller)

    # Verify cameras
    active_cams = []
    for name in CAMERAS:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id >= 0:
            active_cams.append(name)
            print(f"  Camera ready: '{name}'")
        else:
            print(f"  WARNING: camera '{name}' not found – skipping.")

    print(f"\nRunning – {len(active_cams)} camera(s). Press Q to quit.\n")

    renderer     = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    last_print_t = time.time()

    try:
        while True:
            # Stop if GUI was closed
            if not gui.is_running():
                print("GUI closed – shutting down.")
                break

            controller.apply_targets()
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            if PRINT_STATE_EVERY_SEC > 0:
                now = time.time()
                if now - last_print_t >= PRINT_STATE_EVERY_SEC:
                    controller.print_state()
                    last_print_t = now

            # ── Render each camera into a frame ───────────────────────────
            frames = []
            for cam_name in active_cams:
                renderer.update_scene(data, camera=cam_name)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # Camera label overlay
                cv2.putText(bgr, cam_name, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                frames.append(bgr)

            # ── Tile into a 2-column grid ─────────────────────────────────
            n      = len(frames)
            n_cols = min(n, 2)
            n_rows = (n + n_cols - 1) // n_cols
            # Pad with black frames if the grid isn't full
            blank  = np.zeros_like(frames[0])
            while len(frames) < n_rows * n_cols:
                frames.append(blank)
            rows   = []
            for r in range(n_rows):
                row_frames = frames[r * n_cols: (r + 1) * n_cols]
                rows.append(np.hstack(row_frames))
            grid = np.vstack(rows)

            cv2.imshow("Aloha Cameras", grid)

            quit_requested = False
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                quit_requested = True
            try:
                if cv2.getWindowProperty("Aloha Cameras", cv2.WND_PROP_VISIBLE) < 1:
                    quit_requested = True
            except cv2.error:
                quit_requested = True

            if quit_requested:
                print("Quit requested – shutting down.")
                break

    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
