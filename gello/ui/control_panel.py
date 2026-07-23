import queue as _queue_mod
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

import numpy as np

from gello.robots.datc_gripper import DATCGripper

FONT_TITLE  = ("Helvetica", 18, "bold")
FONT_BTN_LG = ("Helvetica", 16, "bold")
FONT_BTN_MD = ("Helvetica", 14, "bold")
FONT_BTN_SM = ("Helvetica", 13)
FONT_LABEL  = ("Helvetica", 12)
FONT_MONO   = ("Courier", 12)
PAD = {"padx": 14, "pady": 8}

BG       = "#1e1e2e"
BG_CARD  = "#2a2a3e"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"


class ControlPanel:
    def __init__(
        self,
        teleop_event: threading.Event,
        gripper: Optional[DATCGripper] = None,
        recorder=None,
        vla_demo_fn: Optional[Callable] = None,
        gello_robot=None,
        gc_xml_path: Optional[str] = None,
        gc_torque_to_pwm=None,
        get_gello_joints_fn: Optional[Callable] = None,
        get_ur_joints_fn: Optional[Callable] = None,
        estop_fn: Optional[Callable] = None,
        stop_fn: Optional[Callable] = None,
        go_home_fn: Optional[Callable] = None,
        admittance_on_fn: Optional[Callable] = None,
        admittance_off_fn: Optional[Callable] = None,
        cameras: Optional[Dict] = None,
        record_queue=None,
    ):
        self._teleop_event = teleop_event
        self._gripper = gripper
        self._recorder = recorder
        self._vla_demo_fn = vla_demo_fn
        self._gello_robot = gello_robot
        self._gc_xml_path = gc_xml_path
        self._gc_torque_to_pwm = gc_torque_to_pwm
        self._gc_on = False
        self._get_gello_joints = get_gello_joints_fn
        self._get_ur_joints = get_ur_joints_fn
        self._estop_fn = estop_fn
        self._stop_fn = stop_fn
        self._go_home_fn = go_home_fn
        self._admittance_on_fn = admittance_on_fn
        self._admittance_off_fn = admittance_off_fn
        self._admittance_on = False
        self._cameras = cameras or {}
        self._record_queue = record_queue
        self._impedance_on = False
        self._ui_queue = _queue_mod.Queue()  # 스레드→메인 UI 업데이트 큐

        self._root = tk.Tk()
        self._root.title("Robot Control Panel")
        self._root.resizable(True, True)
        self._root.configure(bg=BG)
        self._build_ui()
        self._start_updates()
        self._poll_ui_queue()

    def _build_ui(self):
        s = ttk.Style()
        s.configure("Card.TLabelframe", background=BG_CARD, padding=10)
        s.configure("Card.TLabelframe.Label", background=BG_CARD, foreground=FG, font=FONT_BTN_SM)

        def section(parent, text):
            f = ttk.LabelFrame(parent, text=text, style="Card.TLabelframe")
            f.pack(fill="x", padx=10, pady=5)
            return f

        # ── 최상단 STOP / E-STOP ──────────────────────────────────────
        top_btn_row = tk.Frame(self._root, bg=BG)
        top_btn_row.pack(fill="x", padx=16, pady=(14, 6))
        tk.Button(
            top_btn_row, text="■  STOP",
            bg="#e67e22", fg="white",
            font=("Helvetica", 22, "bold"),
            relief="raised", bd=5, height=2,
            command=self._stop,
        ).pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Button(
            top_btn_row, text="⛔  E-STOP",
            bg="#c0392b", fg="white",
            font=("Helvetica", 22, "bold"),
            relief="raised", bd=5, height=2,
            command=self._estop,
        ).pack(side="left", fill="both", expand=True, padx=(6, 0))

        # ── 2열 레이아웃 ──────────────────────────────────────────────
        body = tk.Frame(self._root, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        left  = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y", padx=4)
        right.pack(side="left", fill="both", expand=True, padx=4)

        # ── 왼쪽: 컨트롤 ─────────────────────────────────────────────

        # Teleoperation
        teleop_frame = ttk.LabelFrame(left, text="Teleoperation", style="Card.TLabelframe")
        teleop_frame.pack(fill="x", padx=4, pady=5)
        row = tk.Frame(teleop_frame, bg=BG_CARD)
        row.pack(fill="x", pady=4)
        self._teleop_btn = tk.Button(
            row, text="○  Teleop OFF",
            bg="#e74c3c", fg="white", font=FONT_BTN_LG,
            relief="flat", width=16, height=2, command=self._toggle_teleop,
        )
        self._teleop_btn.pack(side="left", padx=(6, 4), pady=4)
        tk.Button(
            row, text="🏠  Go Home",
            bg="#2980b9", fg="white", font=FONT_BTN_LG,
            relief="flat", width=14, height=2, command=self._go_home,
        ).pack(side="left", padx=(4, 6), pady=4)

        # Data Recording
        rec_frame = ttk.LabelFrame(left, text="Data Recording", style="Card.TLabelframe")
        rec_frame.pack(fill="x", padx=4, pady=5)
        rec_row = tk.Frame(rec_frame, bg=BG_CARD)
        rec_row.pack(fill="x", pady=4)
        self._rec_btn = tk.Button(
            rec_row, text="● Record",
            bg="#2ecc71", fg="white", font=FONT_BTN_MD,
            relief="flat", width=14, height=2, command=self._toggle_record,
        )
        self._rec_btn.pack(side="left", padx=(6, 4), pady=4)
        tk.Button(
            rec_row, text="Discard",
            bg="#e67e22", fg="white", font=FONT_BTN_MD,
            relief="flat", width=10, height=2, command=self._discard_episode,
        ).pack(side="left", padx=(4, 6), pady=4)
        self._rec_status = tk.Label(rec_frame, text="", font=FONT_LABEL, bg=BG_CARD, fg="#a6e3a1")
        self._rec_status.pack(pady=(0, 4))

        # VLA
        vla_frame = ttk.LabelFrame(left, text="VLA", style="Card.TLabelframe")
        vla_frame.pack(fill="x", padx=4, pady=5)
        tk.Button(
            vla_frame, text="▶  VLA Demo",
            bg="#8e44ad", fg="white", font=FONT_BTN_MD,
            relief="flat", width=20, height=2, command=self._vla_demo,
        ).pack(padx=10, pady=8)

        # Gravity Compensation
        gc_frame = ttk.LabelFrame(left, text="Gravity Compensation", style="Card.TLabelframe")
        gc_frame.pack(fill="x", padx=4, pady=5)
        self._gc_btn = tk.Button(
            gc_frame, text="○  GravComp OFF",
            bg="#e74c3c", fg="white", font=FONT_BTN_MD,
            relief="flat", width=20, height=2, command=self._toggle_gc,
        )
        self._gc_btn.pack(padx=10, pady=8)

        self._adm_btn = tk.Button(
            gc_frame, text="○  Admittance OFF",
            bg="#e74c3c", fg="white", font=FONT_BTN_MD,
            relief="flat", width=20, height=2, command=self._toggle_admittance,
        )
        self._adm_btn.pack(padx=10, pady=(0, 8))

        # Gripper
        gripper_frame = ttk.LabelFrame(left, text="Gripper", style="Card.TLabelframe")
        gripper_frame.pack(fill="x", padx=4, pady=5)
        g_row1 = tk.Frame(gripper_frame, bg=BG_CARD)
        g_row1.pack(fill="x", pady=4)
        tk.Button(g_row1, text="OPEN", bg="#3498db", fg="white", font=FONT_BTN_MD,
                  relief="flat", width=10, height=2, command=self._gripper_open,
                  ).pack(side="left", padx=(6, 4), pady=4)
        tk.Button(g_row1, text="CLOSE", bg="#e74c3c", fg="white", font=FONT_BTN_MD,
                  relief="flat", width=10, height=2, command=self._gripper_close,
                  ).pack(side="left", padx=(4, 6), pady=4)
        g_row2 = tk.Frame(gripper_frame, bg=BG_CARD)
        g_row2.pack(fill="x", padx=8, pady=4)
        tk.Label(g_row2, text="Position (0–1000):", font=FONT_LABEL, bg=BG_CARD, fg=FG).pack(side="left")
        self._pos_var = tk.StringVar(value="500")
        tk.Entry(g_row2, textvariable=self._pos_var, width=8, font=FONT_LABEL).pack(side="left", padx=6)
        tk.Button(g_row2, text="Set", command=self._set_position,
                  font=FONT_BTN_SM, bg="#45475a", fg="white", relief="flat", width=6).pack(side="left")
        g_row3 = tk.Frame(gripper_frame, bg=BG_CARD)
        g_row3.pack(fill="x", padx=8, pady=(4, 8))
        tk.Label(g_row3, text="Impedance:", font=FONT_LABEL, bg=BG_CARD, fg=FG).pack(side="left")
        self._imp_btn = tk.Button(
            g_row3, text="OFF", width=8, bg="#e74c3c", fg="white",
            font=FONT_BTN_SM, relief="flat", command=self._toggle_impedance,
        )
        self._imp_btn.pack(side="left", padx=6)

        # ── 오른쪽: 관절각 + 카메라 ──────────────────────────────────

        # Joint angles table
        joint_frame = ttk.LabelFrame(right, text="Joint State (degrees)", style="Card.TLabelframe")
        joint_frame.pack(fill="x", padx=4, pady=5)

        header = tk.Frame(joint_frame, bg=BG_CARD)
        header.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(header, text="Joint", width=6,  font=FONT_MONO, bg=BG_CARD, fg=FG_DIM, anchor="center").pack(side="left")
        tk.Label(header, text="UR10",  width=10, font=FONT_MONO, bg=BG_CARD, fg="#89b4fa", anchor="center").pack(side="left")
        tk.Label(header, text="GELLO", width=10, font=FONT_MONO, bg=BG_CARD, fg="#a6e3a1", anchor="center").pack(side="left")
        tk.Label(header, text="Diff",  width=10, font=FONT_MONO, bg=BG_CARD, fg="#f38ba8", anchor="center").pack(side="left")

        self._joint_rows = []
        for i in range(6):
            row = tk.Frame(joint_frame, bg=BG_CARD)
            row.pack(fill="x", padx=6, pady=1)
            lbl_j    = tk.Label(row, text=f"J{i+1}", width=6,  font=FONT_MONO, bg=BG_CARD, fg=FG_DIM,    anchor="center")
            lbl_ur   = tk.Label(row, text="--",      width=10, font=FONT_MONO, bg=BG_CARD, fg="#89b4fa",  anchor="center")
            lbl_gello= tk.Label(row, text="--",      width=10, font=FONT_MONO, bg=BG_CARD, fg="#a6e3a1",  anchor="center")
            lbl_diff = tk.Label(row, text="--",      width=10, font=FONT_MONO, bg=BG_CARD, fg="#f38ba8",  anchor="center")
            lbl_j.pack(side="left"); lbl_ur.pack(side="left")
            lbl_gello.pack(side="left"); lbl_diff.pack(side="left")
            self._joint_rows.append((lbl_ur, lbl_gello, lbl_diff))

        tk.Frame(joint_frame, bg=BG_CARD, height=4).pack()

        # Camera streams
        if self._cameras:
            cam_frame = ttk.LabelFrame(right, text="Camera", style="Card.TLabelframe")
            cam_frame.pack(fill="both", expand=True, padx=4, pady=5)
            cam_row = tk.Frame(cam_frame, bg=BG_CARD)
            cam_row.pack(fill="both", expand=True, pady=4)
            self._cam_labels = {}
            # wrist가 있고 exterior가 없으면 detect 슬롯 추가 (검출 오버레이 표시용)
            display_names = list(self._cameras.keys())
            if "wrist" in self._cameras and "exterior" not in self._cameras:
                display_names.append("detect")
            for name in display_names:
                col = tk.Frame(cam_row, bg=BG_CARD)
                col.pack(side="left", padx=6, expand=True)
                tk.Label(col, text=name, font=FONT_LABEL, bg=BG_CARD, fg=FG_DIM).pack()
                lbl = tk.Label(col, bg="#000000")
                lbl.pack()
                self._cam_labels[name] = lbl
        else:
            self._cam_labels = {}

    def _poll_ui_queue(self):
        """메인 스레드에서 주기적으로 UI 업데이트 큐를 처리."""
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                fn()
        except _queue_mod.Empty:
            pass
        self._root.after(50, self._poll_ui_queue)

    def _schedule_ui(self, fn):
        """백그라운드 스레드에서 안전하게 UI 업데이트를 예약."""
        self._ui_queue.put(fn)

    # ── Periodic update ──────────────────────────────────────────────

    def _start_updates(self):
        self._update_joints()
        self._update_cameras()

    def _update_joints(self):
        def _read():
            try:
                ur_j    = np.rad2deg(np.array(self._get_ur_joints()))    if self._get_ur_joints    else None
                gello_j = np.rad2deg(np.array(self._get_gello_joints())) if self._get_gello_joints else None
            except Exception:
                ur_j, gello_j = None, None
            def _apply():
                try:
                    for i, (lbl_ur, lbl_gello, lbl_diff) in enumerate(self._joint_rows):
                        ur_v    = f"{ur_j[i]:+7.1f}°"    if ur_j    is not None and i < len(ur_j)    else "--"
                        gello_v = f"{gello_j[i]:+7.1f}°" if gello_j is not None and i < len(gello_j) else "--"
                        if ur_j is not None and gello_j is not None and i < len(ur_j) and i < len(gello_j):
                            diff = abs(ur_j[i] - gello_j[i])
                            diff_v = f"{diff:6.1f}°"
                            lbl_diff.config(fg="#f38ba8" if diff >= 30 else "#a6e3a1")
                        else:
                            diff_v = "--"
                        lbl_ur.config(text=ur_v)
                        lbl_gello.config(text=gello_v)
                        lbl_diff.config(text=diff_v)
                except Exception:
                    pass
            self._schedule_ui(_apply)
        threading.Thread(target=_read, daemon=True).start()
        self._root.after(100, self._update_joints)  # 10Hz

    def _update_cameras(self):
        if not self._cam_labels:
            return
        # 카메라 I/O는 백그라운드, PIL→PhotoImage 변환은 메인 스레드에서 수행
        def _read():
            pil_images = {}
            try:
                import cv2 as _cv2
                import numpy as _np
                from PIL import Image

                # wrist 원본을 먼저 읽어두기 (exterior 검출 오버레이에도 사용)
                wrist_cam = self._cameras.get("wrist")
                wrist_rgb = None
                if wrist_cam is not None:
                    wrist_rgb, _ = wrist_cam.read()

                for name in self._cam_labels:
                    if name == "wrist":
                        if wrist_rgb is not None:
                            pil_images[name] = Image.fromarray(wrist_rgb).resize((320, 240))

                    elif name in ("exterior", "detect"):
                        # exterior 슬롯: wrist 카메라 검출 오버레이 (preview_color_detect.py 방식)
                        if wrist_rgb is None:
                            continue
                        bgr = _cv2.cvtColor(wrist_rgb, _cv2.COLOR_RGB2BGR)
                        vis = bgr.copy()
                        h, w = bgr.shape[:2]
                        hsv = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2HSV)

                        # 노랑 마스크 → cyan overlay (좌우 15% 배제)
                        mask_y = _cv2.inRange(hsv, _np.array([22, 150, 120]), _np.array([32, 255, 255]))
                        mask_y = _cv2.erode(mask_y, None, iterations=2)
                        mask_y = _cv2.dilate(mask_y, None, iterations=2)
                        y_left  = int(w * 0.15)
                        y_right = int(w * 0.85)
                        mask_y[:, :y_left]  = 0
                        mask_y[:, y_right:] = 0
                        vis[mask_y > 0] = (0, 200, 200)
                        _cv2.rectangle(vis, (y_left, 0), (y_right, h - 1), (0, 200, 200), 1)

                        # 케이블(검정) 마스크 → blue overlay, ROI 내부만
                        top_y   = int(h * 0.51)
                        bot_y   = int(h * 0.65)
                        left_x  = int(w * 0.448)
                        right_x = int(w * 0.593)
                        mask_c = _cv2.inRange(hsv, _np.array([0, 0, 0]), _np.array([180, 80, 80]))
                        mask_c = _cv2.erode(mask_c, None, iterations=2)
                        mask_c = _cv2.dilate(mask_c, None, iterations=2)
                        roi_mask = _np.zeros_like(mask_c)
                        roi_mask[top_y:bot_y, left_x:right_x] = mask_c[top_y:bot_y, left_x:right_x]
                        vis[roi_mask > 0] = (200, 200, 0)

                        # ROI 사각형 (빨강)
                        _cv2.rectangle(vis, (left_x, top_y), (right_x, bot_y), (0, 0, 255), 1)

                        # 노랑 중심점
                        pts_y = _np.argwhere(mask_y > 0)
                        if len(pts_y) >= 20:
                            cy_y = int(pts_y[:, 0].mean())
                            cx_y = int(pts_y[:, 1].mean())
                            _cv2.circle(vis, (cx_y, cy_y), 8, (0, 255, 255), -1)
                            _cv2.putText(vis, "yellow", (cx_y + 10, cy_y),
                                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # 케이블 tip 중심점 (ROI 하단 20px 스트립)
                        contours, _ = _cv2.findContours(roi_mask, _cv2.RETR_EXTERNAL,
                                                        _cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=_cv2.contourArea)
                            if _cv2.contourArea(c) >= 200:
                                strip_top = max(0, bot_y - 20)
                                strip = roi_mask[strip_top:bot_y, :]
                                cols = _np.where(strip.any(axis=0))[0]
                                if len(cols) > 0:
                                    cx_c = int(cols.mean())
                                    cy_c = bot_y - 10
                                    _cv2.circle(vis, (cx_c, cy_c), 8, (255, 200, 0), -1)
                                    _cv2.putText(vis, "cable", (cx_c + 10, cy_c),
                                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                        vis_rgb = _cv2.cvtColor(vis, _cv2.COLOR_BGR2RGB)
                        pil_images[name] = Image.fromarray(vis_rgb).resize((320, 240))

                    else:
                        cam = self._cameras.get(name)
                        if cam is not None:
                            img, _ = cam.read()
                            pil_images[name] = Image.fromarray(img).resize((320, 240))

            except Exception as _e:
                import traceback as _tb
                print(f"[Camera] 읽기 오류: {_e}")
                _tb.print_exc()

            def _update():
                try:
                    from PIL import ImageTk
                    for name, pil in pil_images.items():
                        lbl = self._cam_labels.get(name)
                        if lbl is None:
                            continue
                        photo = ImageTk.PhotoImage(pil)
                        lbl.configure(image=photo)
                        lbl.image = photo
                except Exception as _e:
                    import traceback as _tb
                    print(f"[Camera UI] 업데이트 오류: {_e}")
                    _tb.print_exc()
            self._schedule_ui(_update)
        threading.Thread(target=_read, daemon=True).start()
        self._root.after(66, self._update_cameras)  # ~15Hz

    # ── Callbacks ────────────────────────────────────────────────────

    def _stop(self):
        self._teleop_event.clear()
        self._teleop_btn.config(text="○  Teleop OFF", bg="#e74c3c")
        if self._stop_fn:
            threading.Thread(target=self._stop_fn, daemon=True).start()
        print("\n[STOP] VLA/Teleop stopped.\n")

    def _estop(self):
        self._teleop_event.clear()
        self._teleop_btn.config(text="○  Teleop OFF", bg="#e74c3c")
        if self._estop_fn:
            threading.Thread(target=self._estop_fn, daemon=True).start()
        print("\n[E-STOP] UR robot stopped. Teleoperation disabled.\n")

    def _toggle_teleop(self):
        if self._teleop_event.is_set():
            self._teleop_event.clear()
            self._teleop_btn.config(text="○  Teleop OFF", bg="#e74c3c")
        else:
            if self._get_gello_joints is not None and self._get_ur_joints is not None:
                try:
                    _gj = self._get_gello_joints()
                    _uj = self._get_ur_joints()
                    if _gj is None or _uj is None:
                        raise ValueError("joint cache not ready yet")
                    gello = np.array(_gj)
                    ur    = np.array(_uj)
                    n = min(len(gello), len(ur))
                    diff_deg = np.abs(np.rad2deg(gello[:n] - ur[:n]))
                    over = np.where(diff_deg >= 30.0)[0]
                    if len(over) > 0:
                        print("[Teleop] 시작 거부: 관절 차이가 30도 이상인 축이 있습니다.")
                        for idx in over:
                            print(f"  Joint {idx+1}: GELLO={np.rad2deg(gello[idx]):.1f}°, "
                                  f"UR={np.rad2deg(ur[idx]):.1f}°, diff={diff_deg[idx]:.1f}°")
                        return
                except Exception as e:
                    print(f"[Teleop] 관절 비교 중 오류 (무시하고 시작): {e}")
            self._teleop_event.set()
            self._teleop_btn.config(text="●  Teleop ON", bg="#2ecc71")

    def _go_home(self):
        if self._go_home_fn is None:
            return
        self._teleop_event.clear()
        self._teleop_btn.config(text="○  Teleop OFF", bg="#e74c3c")
        threading.Thread(target=self._go_home_fn, daemon=True).start()

    def _toggle_record(self):
        if self._recorder is None:
            return
        if not self._recorder.is_recording:
            self._recorder.start_episode()
            self._rec_btn.config(text="■  Stop", bg="#e74c3c")
            self._rec_status.config(text=f"Recording episode {self._recorder._episode_count}...")
        else:
            self._rec_btn.config(text="Saving...", bg="#95a5a6", state="disabled")
            self._rec_status.config(text="저장 중...")
            def _save():
                status_text = "Save error: unknown"
                try:
                    self._recorder.end_episode(save=True, record_queue=self._record_queue)
                    count = self._recorder._episode_count
                    status_text = f"Saved. Total: {count} episodes"
                except Exception as e:
                    import traceback; traceback.print_exc()
                    status_text = f"Save error: {e}"
                print(f"[UI] Scheduling button re-enable, status: {status_text}")
                _st = status_text
                def _update_ui(st=_st):
                    print("[UI] _update_ui called")
                    self._rec_btn.config(text="● Record", bg="#2ecc71", state="normal")
                    self._rec_status.config(text=st)
                self._schedule_ui(_update_ui)
            threading.Thread(target=_save, daemon=True).start()

    def _discard_episode(self):
        if self._recorder and self._recorder.is_recording:
            self._recorder.end_episode(save=False)
            self._rec_btn.config(text="● Record", bg="#2ecc71")
            self._rec_status.config(text="Episode discarded.")

    def _toggle_gc(self):
        if self._gello_robot is None:
            return
        was_teleop_on = self._teleop_event.is_set()
        self._teleop_event.clear()
        self._gc_btn.config(state="disabled")
        turning_on = not self._gc_on

        def _do_gc():
            import time
            time.sleep(0.3)  # 현재 control loop 사이클 완료 대기
            try:
                if turning_on:
                    self._gello_robot.start_gravity_compensation(
                        xml_path=self._gc_xml_path, torque_to_pwm=self._gc_torque_to_pwm,
                    )
                    self._gc_on = True
                    self._root.after(0, lambda: self._gc_btn.config(
                        text="●  GravComp ON", bg="#2ecc71", state="normal"))
                else:
                    self._gello_robot.stop_gravity_compensation()
                    self._gc_on = False
                    self._root.after(0, lambda: self._gc_btn.config(
                        text="○  GravComp OFF", bg="#e74c3c", state="normal"))
                if was_teleop_on:
                    time.sleep(0.2)
                    self._teleop_event.set()
            except Exception as e:
                print(f"[GravComp] 토글 오류: {e}")
                # 실패 시 상태 롤백
                label = "○  GravComp OFF" if turning_on else "●  GravComp ON"
                color = "#e74c3c" if turning_on else "#2ecc71"
                self._root.after(0, lambda: self._gc_btn.config(
                    text=label, bg=color, state="normal"))

        threading.Thread(target=_do_gc, daemon=True).start()

    def _vla_demo(self):
        if self._vla_demo_fn:
            threading.Thread(target=self._vla_demo_fn, daemon=True).start()

    def _gripper_open(self):
        if self._gripper: self._gripper.open()

    def _gripper_close(self):
        if self._gripper: self._gripper.close()

    def _set_position(self):
        if self._gripper:
            try:
                self._gripper.set_position(int(self._pos_var.get()))
            except ValueError:
                pass

    def _toggle_admittance(self):
        turning_on = not self._admittance_on
        fn = self._admittance_on_fn if turning_on else self._admittance_off_fn
        if fn is None:
            return
        try:
            fn()
            self._admittance_on = turning_on
            if turning_on:
                self._adm_btn.config(text="●  Admittance ON", bg="#2ecc71")
            else:
                self._adm_btn.config(text="○  Admittance OFF", bg="#e74c3c")
        except Exception as e:
            print(f"[Admittance] 오류: {e}")

    def _toggle_impedance(self):
        if self._gripper:
            if self._impedance_on:
                self._gripper.impedance_off()
                self._imp_btn.config(text="OFF", bg="#e74c3c")
            else:
                self._gripper.impedance_on()
                self._imp_btn.config(text="ON", bg="#2ecc71")
            self._impedance_on = not self._impedance_on

    def run(self):
        self._root.mainloop()
