import queue as _queue_mod
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from gello.robots.datc_gripper import DATCGripper


def _scan_checkpoints(checkpoints_dir: str) -> List[Tuple[str, str]]:
    """checkpoints_dir/<version>/checkpoints/<step>/pretrained_model 구조를 스캔.
    반환: [(표시용 라벨 "version/step", pretrained_model 경로), ...]"""
    root = Path(checkpoints_dir).expanduser()
    found = []
    if not root.is_dir():
        return found
    for version_dir in sorted(root.iterdir()):
        steps_dir = version_dir / "checkpoints"
        if not steps_dir.is_dir():
            continue
        for step_dir in sorted(steps_dir.iterdir()):
            pm = step_dir / "pretrained_model"
            if pm.is_dir():
                found.append((f"{version_dir.name}/{step_dir.name}", str(pm)))
    return found


def _scan_datasets(datasets_dir: str) -> List[str]:
    """datasets_dir 아래 데이터셋 디렉터리 이름 목록 (백업 사본은 제외)."""
    root = Path(datasets_dir).expanduser()
    found = []
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        if d.is_dir() and "_backup_" not in d.name:
            found.append(d.name)
    return found


# (train_act.py 인자명, 캐스팅 함수, 기본값) — ACT 학습 파라미터
_ACT_TRAIN_PARAMS = [
    ("batch_size", int, 8),
    ("num_workers", int, 4),
    ("steps", int, 100000),
    ("save_freq", int, 5000),
    ("lr", float, 1e-4),
    ("chunk_size", int, 50),
]

# (train_align.py 인자명, 캐스팅 함수, 기본값) — Align 학습 파라미터
_ALIGN_TRAIN_PARAMS = [
    ("epochs", int, 200),
    ("batch_size", int, 256),
    ("lr", float, 1e-3),
    ("fine_threshold", float, 0.007),
]

# (yaml 키, 캐스팅 함수, 기본값) — VLA Parameters 패널에 노출할 수치형 파라미터
_VLA_NUMERIC_PARAMS = [
    ("speed_scale", float, 1.0),
    ("delta_scale", float, 1.0),
    ("chunk_size", int, 20),
    ("z_insert", float, 0.24),
]
# (yaml 키, 기본값, 표시 라벨) — 체크박스로 노출할 불리언 파라미터
_VLA_BOOL_PARAMS = [
    ("control_gripper", True, "control_gripper"),
    ("finger_change", True, "Finger Change"),
]
# (yaml 키, 선택지, 기본값) — 드롭다운으로 노출할 문자열/다중값 파라미터
_VLA_CHOICE_PARAMS = [
    ("use_z_freeze", ["false", "act", "servoing"], "false"),
]

# 맑은 고딕은 Windows 전용 폰트라 이 리눅스 환경엔 없어 동일 계열의 Noto Sans CJK KR로 대체
FONT_FAMILY      = "Noto Sans CJK KR"
FONT_MONO_FAMILY = "Noto Sans Mono CJK KR"
FONT_TITLE  = (FONT_FAMILY, 18, "bold")
FONT_BTN_LG = (FONT_FAMILY, 16, "bold")
FONT_BTN_MD = (FONT_FAMILY, 14, "bold")
FONT_BTN_SM = (FONT_FAMILY, 13, "bold")
FONT_LABEL  = (FONT_FAMILY, 12, "bold")
FONT_MONO   = (FONT_MONO_FAMILY, 12, "bold")
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
        recorder_lock: Optional[threading.Lock] = None,
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
        checkpoints_dir: str = "~/checkpoints",
        datasets_dir: str = "~/datasets",
        current_checkpoint: Optional[str] = None,
        current_dataset: Optional[str] = None,
        set_checkpoint_fn: Optional[Callable[[str], None]] = None,
        set_dataset_fn: Optional[Callable[[str], None]] = None,
        set_unwrap_rotvec_fn: Optional[Callable[[bool], None]] = None,
        current_unwrap_rotvec: bool = False,
        vla_params: Optional[Dict] = None,
        set_vla_params_fn: Optional[Callable[[Dict], None]] = None,
        start_act_training_fn: Optional[Callable[[Dict], None]] = None,
        start_align_training_fn: Optional[Callable[[Dict], None]] = None,
    ):
        self._teleop_event = teleop_event
        self._gripper = gripper
        self._recorder = recorder
        self._recorder_lock = recorder_lock
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
        self._saving = False  # 녹화 저장(record_queue 드레인) 도중 Go Home 등 로봇 명령 충돌 방지
        self._ui_queue = _queue_mod.Queue()  # 스레드→메인 UI 업데이트 큐

        self._checkpoints_dir = checkpoints_dir
        self._datasets_dir = datasets_dir
        self._set_checkpoint_fn = set_checkpoint_fn
        self._set_dataset_fn = set_dataset_fn
        self._set_unwrap_rotvec_fn = set_unwrap_rotvec_fn
        self._checkpoint_map: Dict[str, str] = {}  # 라벨 → pretrained_model 경로
        self._current_checkpoint_label = current_checkpoint
        self._current_dataset = current_dataset
        self._current_unwrap_rotvec = current_unwrap_rotvec
        self._vla_params = vla_params or {}
        self._set_vla_params_fn = set_vla_params_fn
        self._start_act_training_fn = start_act_training_fn
        self._start_align_training_fn = start_align_training_fn

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
            font=(FONT_FAMILY, 22, "bold"),
            relief="raised", bd=5, height=2,
            command=self._stop,
        ).pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Button(
            top_btn_row, text="⛔  E-STOP",
            bg="#c0392b", fg="white",
            font=(FONT_FAMILY, 22, "bold"),
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

        # Model / Dataset selection
        md_frame = ttk.LabelFrame(left, text="Model / Dataset", style="Card.TLabelframe")
        md_frame.pack(fill="x", padx=4, pady=5)

        ckpt_row = tk.Frame(md_frame, bg=BG_CARD)
        ckpt_row.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(ckpt_row, text="Checkpoint:", width=10, font=FONT_LABEL, bg=BG_CARD, fg=FG, anchor="w").pack(side="left")
        self._ckpt_var = tk.StringVar()
        self._ckpt_combo = ttk.Combobox(ckpt_row, textvariable=self._ckpt_var, state="readonly", width=22)
        self._ckpt_combo.pack(side="left", padx=6)
        self._ckpt_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selected)
        tk.Button(ckpt_row, text="↻", width=3, font=FONT_BTN_SM, bg="#45475a", fg="white",
                  relief="flat", command=self._refresh_checkpoints).pack(side="left")

        ds_row = tk.Frame(md_frame, bg=BG_CARD)
        ds_row.pack(fill="x", padx=8, pady=(2, 6))
        tk.Label(ds_row, text="Dataset:", width=10, font=FONT_LABEL, bg=BG_CARD, fg=FG, anchor="w").pack(side="left")
        self._ds_var = tk.StringVar()
        self._ds_combo = ttk.Combobox(ds_row, textvariable=self._ds_var, state="readonly", width=22)
        self._ds_combo.pack(side="left", padx=6)
        self._ds_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        tk.Button(ds_row, text="↻", width=3, font=FONT_BTN_SM, bg="#45475a", fg="white",
                  relief="flat", command=self._refresh_datasets).pack(side="left")

        self._md_status = tk.Label(md_frame, text="", font=FONT_LABEL, bg=BG_CARD, fg="#a6e3a1", justify="left")
        self._md_status.pack(padx=8, pady=(0, 6), anchor="w")

        orient_row = tk.Frame(md_frame, bg=BG_CARD)
        orient_row.pack(fill="x", padx=8, pady=(0, 6))
        self._unwrap_rotvec_var = tk.BooleanVar(value=bool(self._current_unwrap_rotvec))
        tk.Checkbutton(
            orient_row, text="unwrap_rotvec (체크=v20 축각 / 해제=v19 RPY)",
            variable=self._unwrap_rotvec_var, font=FONT_BTN_SM,
            bg=BG_CARD, fg=FG, selectcolor=BG_CARD, activebackground=BG_CARD,
            command=self._toggle_unwrap_rotvec,
        ).pack(side="left")

        self._refresh_checkpoints()
        self._refresh_datasets()

        # VLA
        vla_frame = ttk.LabelFrame(left, text="VLA", style="Card.TLabelframe")
        vla_frame.pack(fill="x", padx=4, pady=5)
        tk.Button(
            vla_frame, text="▶  VLA Demo",
            bg="#8e44ad", fg="white", font=FONT_BTN_MD,
            relief="flat", width=20, height=2, command=self._vla_demo,
        ).pack(padx=10, pady=8)

        # VLA Parameters
        self._vla_param_vars: Dict[str, tk.StringVar] = {}
        self._vla_bool_vars: Dict[str, tk.BooleanVar] = {}
        param_grid = tk.Frame(vla_frame, bg=BG_CARD)
        param_grid.pack(fill="x", padx=8, pady=(0, 4))
        for i, (key, cast, default) in enumerate(_VLA_NUMERIC_PARAMS):
            r, c = divmod(i, 2)
            cell = tk.Frame(param_grid, bg=BG_CARD)
            cell.grid(row=r, column=c, sticky="w", padx=4, pady=2)
            tk.Label(cell, text=f"{key}:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(self._vla_params.get(key, default)))
            tk.Entry(cell, textvariable=var, width=8, font=FONT_BTN_SM).pack(side="left")
            self._vla_param_vars[key] = var

        bool_row = tk.Frame(vla_frame, bg=BG_CARD)
        bool_row.pack(fill="x", padx=8, pady=(0, 4))
        for key, default, label in _VLA_BOOL_PARAMS:
            var = tk.BooleanVar(value=bool(self._vla_params.get(key, default)))
            tk.Checkbutton(
                bool_row, text=label, variable=var, font=FONT_BTN_SM,
                bg=BG_CARD, fg=FG, selectcolor=BG_CARD, activebackground=BG_CARD,
            ).pack(side="left", padx=4)
            self._vla_bool_vars[key] = var

        self._vla_choice_vars: Dict[str, tk.StringVar] = {}
        choice_row = tk.Frame(vla_frame, bg=BG_CARD)
        choice_row.pack(fill="x", padx=8, pady=(0, 4))
        for key, choices, default in _VLA_CHOICE_PARAMS:
            cell = tk.Frame(choice_row, bg=BG_CARD)
            cell.pack(side="left", padx=4)
            tk.Label(cell, text=f"{key}:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG).pack(side="left")
            current = str(self._vla_params.get(key, default))
            if current not in choices:
                current = default
            var = tk.StringVar(value=current)
            ttk.Combobox(
                cell, textvariable=var, values=choices, state="readonly", width=10,
            ).pack(side="left", padx=(4, 0))
            self._vla_choice_vars[key] = var

        self._vla_param_status = tk.Label(vla_frame, text="", font=FONT_BTN_SM, bg=BG_CARD, fg="#a6e3a1")
        self._vla_param_status.pack(padx=8, pady=(0, 2))
        tk.Button(
            vla_frame, text="Apply Parameters", bg="#45475a", fg="white", font=FONT_BTN_SM,
            relief="flat", command=self._apply_vla_params,
        ).pack(padx=10, pady=(0, 8))

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

        # Training (수집된 데이터셋 기준 오프라인 학습 — 별도 프로세스로 실행)
        train_frame = ttk.LabelFrame(left, text="Training (offline)", style="Card.TLabelframe")
        train_frame.pack(fill="x", padx=4, pady=5)
        tk.Label(
            train_frame, text="학습 대상 데이터셋 = 위 Model / Dataset에서 선택된 데이터셋",
            font=FONT_BTN_SM, bg=BG_CARD, fg=FG_DIM, wraplength=280, justify="left",
        ).pack(padx=8, pady=(6, 2), anchor="w")

        # ACT
        act_sub = tk.LabelFrame(train_frame, text="ACT", bg=BG_CARD, fg=FG, font=FONT_BTN_SM)
        act_sub.pack(fill="x", padx=8, pady=4)
        self._act_train_vars: Dict[str, tk.StringVar] = {}
        act_grid = tk.Frame(act_sub, bg=BG_CARD)
        act_grid.pack(fill="x", padx=4, pady=2)
        for i, (key, cast, default) in enumerate(_ACT_TRAIN_PARAMS):
            r, c = divmod(i, 2)
            cell = tk.Frame(act_grid, bg=BG_CARD)
            cell.grid(row=r, column=c, sticky="w", padx=4, pady=2)
            tk.Label(cell, text=f"{key}:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            tk.Entry(cell, textvariable=var, width=8, font=FONT_BTN_SM).pack(side="left")
            self._act_train_vars[key] = var
        act_name_row = tk.Frame(act_sub, bg=BG_CARD)
        act_name_row.pack(fill="x", padx=4, pady=2)
        tk.Label(act_name_row, text="output_name:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG, width=12, anchor="w").pack(side="left")
        self._act_output_var = tk.StringVar(value="ur10_act_new")
        tk.Entry(act_name_row, textvariable=self._act_output_var, width=18, font=FONT_BTN_SM).pack(side="left")
        tk.Button(
            act_sub, text="▶  Start ACT Training", bg="#8e44ad", fg="white", font=FONT_BTN_SM,
            relief="flat", command=self._start_act_training,
        ).pack(padx=4, pady=(2, 6))

        # Align
        align_sub = tk.LabelFrame(train_frame, text="Align", bg=BG_CARD, fg=FG, font=FONT_BTN_SM)
        align_sub.pack(fill="x", padx=8, pady=(0, 4))
        self._align_train_vars: Dict[str, tk.StringVar] = {}
        align_grid = tk.Frame(align_sub, bg=BG_CARD)
        align_grid.pack(fill="x", padx=4, pady=2)
        for i, (key, cast, default) in enumerate(_ALIGN_TRAIN_PARAMS):
            r, c = divmod(i, 2)
            cell = tk.Frame(align_grid, bg=BG_CARD)
            cell.grid(row=r, column=c, sticky="w", padx=4, pady=2)
            tk.Label(cell, text=f"{key}:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            tk.Entry(cell, textvariable=var, width=8, font=FONT_BTN_SM).pack(side="left")
            self._align_train_vars[key] = var
        align_name_row = tk.Frame(align_sub, bg=BG_CARD)
        align_name_row.pack(fill="x", padx=4, pady=2)
        tk.Label(align_name_row, text="output_name:", font=FONT_BTN_SM, bg=BG_CARD, fg=FG, width=12, anchor="w").pack(side="left")
        self._align_output_var = tk.StringVar(value="align_xy_mlp.pt")
        tk.Entry(align_name_row, textvariable=self._align_output_var, width=18, font=FONT_BTN_SM).pack(side="left")
        tk.Button(
            align_sub, text="▶  Start Align Training", bg="#8e44ad", fg="white", font=FONT_BTN_SM,
            relief="flat", command=self._start_align_training,
        ).pack(padx=4, pady=(2, 6))

        self._train_status = tk.Label(train_frame, text="", font=FONT_BTN_SM, bg=BG_CARD, fg="#a6e3a1", wraplength=280, justify="left")
        self._train_status.pack(padx=8, pady=(0, 6), anchor="w")

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
                            pil_images[name] = Image.fromarray(wrist_rgb).resize((480, 360))

                    elif name in ("exterior", "detect"):
                        # exterior 슬롯: wrist 카메라 검출 오버레이 (preview_color_detect.py 방식)
                        if wrist_rgb is None:
                            continue
                        bgr = _cv2.cvtColor(wrist_rgb, _cv2.COLOR_RGB2BGR)
                        vis = bgr.copy()
                        h, w = bgr.shape[:2]
                        hsv = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2HSV)

                        # 노랑 마스크 → cyan overlay (좌우 15% 배제)
                        # mask_y = _cv2.inRange(hsv, _np.array([22, 150, 120]), _np.array([32, 255, 255]))  # 기존 노란색 기준
                        mask_y = _cv2.inRange(hsv, _np.array([10, 150, 120]), _np.array([26, 255, 255]))
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
                        pil_images[name] = Image.fromarray(vis_rgb).resize((480, 360))

                    else:
                        cam = self._cameras.get(name)
                        if cam is not None:
                            img, _ = cam.read()
                            pil_images[name] = Image.fromarray(img).resize((480, 360))

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
        if getattr(self, "_saving", False):
            print("[GoHome] 녹화 저장 중에는 Go Home을 실행할 수 없습니다 (저장 완료 후 다시 눌러주세요).")
            return
        self._teleop_event.clear()
        self._teleop_btn.config(text="○  Teleop OFF", bg="#e74c3c")
        threading.Thread(target=self._go_home_fn, daemon=True).start()

    def _rec_lock_ctx(self):
        """recorder.add_frame()과 start_episode()/end_episode()가 절대 동시에
        episode_buffer를 건드리지 못하게 하는 락. 없으면 아무 것도 안 하는 컨텍스트."""
        import contextlib
        return self._recorder_lock if self._recorder_lock is not None else contextlib.nullcontext()

    def _toggle_record(self):
        if self._recorder is None:
            return
        if not self._recorder.is_recording:
            with self._rec_lock_ctx():
                self._recorder.start_episode()
            self._rec_btn.config(text="■  Stop", bg="#e74c3c")
            self._rec_status.config(text=f"Recording episode {self._recorder._episode_count}...")
        else:
            self._rec_btn.config(text="Saving...", bg="#95a5a6", state="disabled")
            self._rec_status.config(text="저장 중...")
            self._saving = True
            def _save():
                status_text = "Save error: unknown"
                status_color = "#f38ba8"
                try:
                    with self._rec_lock_ctx():
                        ok = self._recorder.end_episode(save=True, record_queue=self._record_queue)
                    count = self._recorder._episode_count
                    if ok:
                        status_text = f"Saved. Total: {count} episodes"
                        status_color = "#a6e3a1"
                    else:
                        status_text = f"저장 실패 — 터미널 로그 확인 (Total: {count} episodes)"
                except Exception as e:
                    import traceback; traceback.print_exc()
                    status_text = f"Save error: {e}"
                print(f"[UI] Scheduling button re-enable, status: {status_text}")
                self._saving = False
                _st, _color = status_text, status_color
                def _update_ui(st=_st, color=_color):
                    print("[UI] _update_ui called")
                    self._rec_btn.config(text="● Record", bg="#2ecc71", state="normal")
                    self._rec_status.config(text=st, fg=color)
                self._schedule_ui(_update_ui)
            threading.Thread(target=_save, daemon=True).start()

    def _discard_episode(self):
        if self._recorder and self._recorder.is_recording:
            with self._rec_lock_ctx():
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

    def _refresh_checkpoints(self):
        entries = _scan_checkpoints(self._checkpoints_dir)
        self._checkpoint_map = {label: path for label, path in entries}
        self._ckpt_combo["values"] = list(self._checkpoint_map.keys())
        if self._current_checkpoint_label and self._current_checkpoint_label in self._checkpoint_map:
            self._ckpt_var.set(self._current_checkpoint_label)
        self._update_md_status()

    def _refresh_datasets(self):
        names = _scan_datasets(self._datasets_dir)
        self._ds_combo["values"] = names
        if self._current_dataset and self._current_dataset in names:
            self._ds_var.set(self._current_dataset)
        self._update_md_status()

    def _update_md_status(self):
        ckpt = self._current_checkpoint_label or "(미선택)"
        ds = self._current_dataset or "(미선택)"
        self._md_status.config(text=f"Active checkpoint: {ckpt}\nActive dataset: {ds}")

    def _on_checkpoint_selected(self, event=None):
        label = self._ckpt_var.get()
        path = self._checkpoint_map.get(label)
        if path is None or self._set_checkpoint_fn is None:
            return
        try:
            self._set_checkpoint_fn(path)
            self._current_checkpoint_label = label
        except Exception as e:
            print(f"[Model] 체크포인트 변경 오류: {e}")
        self._update_md_status()

    def _on_dataset_selected(self, event=None):
        name = self._ds_var.get()
        if not name or self._set_dataset_fn is None:
            return
        try:
            self._set_dataset_fn(name)
            self._current_dataset = name
        except Exception as e:
            print(f"[Dataset] 변경 오류: {e}")
        self._update_md_status()

    def _apply_vla_params(self):
        new_values = {}
        for key, cast, _default in _VLA_NUMERIC_PARAMS:
            raw = self._vla_param_vars[key].get()
            try:
                new_values[key] = cast(raw)
            except ValueError:
                self._vla_param_status.config(text=f"잘못된 값: {key}={raw}", fg="#f38ba8")
                return
        for key, _default, _label in _VLA_BOOL_PARAMS:
            new_values[key] = self._vla_bool_vars[key].get()
        for key, _choices, _default in _VLA_CHOICE_PARAMS:
            new_values[key] = self._vla_choice_vars[key].get()
        if self._set_vla_params_fn is not None:
            try:
                self._set_vla_params_fn(new_values)
                self._vla_params.update(new_values)
                self._vla_param_status.config(text="적용됨 (다음 VLA Demo부터 반영)", fg="#a6e3a1")
            except Exception as e:
                self._vla_param_status.config(text=f"적용 오류: {e}", fg="#f38ba8")

    def _start_act_training(self):
        if self._start_act_training_fn is None:
            return
        if not self._current_dataset:
            self._train_status.config(text="학습할 데이터셋을 먼저 선택하세요.", fg="#f38ba8")
            return
        params = {"dataset_name": self._current_dataset, "output_name": self._act_output_var.get().strip()}
        for key, cast, _default in _ACT_TRAIN_PARAMS:
            raw = self._act_train_vars[key].get()
            try:
                params[key] = cast(raw)
            except ValueError:
                self._train_status.config(text=f"잘못된 값: {key}={raw}", fg="#f38ba8")
                return
        try:
            self._start_act_training_fn(params)
            self._train_status.config(text=f"ACT 학습 시작: {params['output_name']}", fg="#a6e3a1")
        except Exception as e:
            self._train_status.config(text=f"ACT 학습 시작 오류: {e}", fg="#f38ba8")

    def _start_align_training(self):
        if self._start_align_training_fn is None:
            return
        if not self._current_dataset:
            self._train_status.config(text="학습할 데이터셋을 먼저 선택하세요.", fg="#f38ba8")
            return
        params = {"dataset_name": self._current_dataset, "output_name": self._align_output_var.get().strip()}
        for key, cast, _default in _ALIGN_TRAIN_PARAMS:
            raw = self._align_train_vars[key].get()
            try:
                params[key] = cast(raw)
            except ValueError:
                self._train_status.config(text=f"잘못된 값: {key}={raw}", fg="#f38ba8")
                return
        try:
            self._start_align_training_fn(params)
            self._train_status.config(text=f"Align 학습 시작: {params['output_name']}", fg="#a6e3a1")
        except Exception as e:
            self._train_status.config(text=f"Align 학습 시작 오류: {e}", fg="#f38ba8")

    def set_recorder(self, recorder):
        """데이터셋 전환 시 새 recorder 인스턴스로 교체."""
        self._recorder = recorder

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

    def _toggle_unwrap_rotvec(self):
        enabled = self._unwrap_rotvec_var.get()
        if self._set_unwrap_rotvec_fn is None:
            return
        try:
            self._set_unwrap_rotvec_fn(enabled)
        except Exception as e:
            print(f"[Orientation] 오류: {e}")

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
