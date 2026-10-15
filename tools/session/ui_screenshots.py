"""Regenerate the screenshots in docs/images from the real screens.

Opens the real launcher (with a throw-away profile folder, so nothing in
data/users is touched), walks it through a demo session and saves a picture of
each screen; then renders the live camera view for three poses by pushing their
reference photographs through the same pipeline the trainer runs.

    python tools/session/ui_screenshots.py            -> docs/images/ui-*.png, hud-*.png

Needs a display (the launcher screens are screen captures of the window) and, for
the camera-view pictures, the MediaPipe model.  No webcam is used.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
os.chdir(HERE)

OUT = os.path.join(HERE, "docs", "images")
DEMO_NAME = "Asha"
HUD_POSES = ("vrikshasana", "utkatasana", "adho_mukha")


def launcher_shots() -> list[str]:
    from PIL import ImageGrab

    import launcher
    from yoga import profile as prof
    from yoga import routines as rt

    prof.USERS_DIR = tempfile.mkdtemp()
    app = launcher.App()
    app.attributes("-topmost", True)
    w, h = launcher.ui.px(1180), launcher.ui.px(800)
    app.geometry(f"{w}x{h}+10+10")
    written: list[str] = []

    def settle(n: int = 8) -> None:
        for _ in range(n):
            app.update_idletasks()
            app.update()
            time.sleep(0.05)

    def shot(name: str) -> None:
        settle()
        app.lift()
        settle(3)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        path = os.path.join(OUT, f"ui-{name}.png")
        ImageGrab.grab((x, y, x + app.winfo_width(), y + app.winfo_height())).save(path)
        written.append(path)

    try:
        shot("1-profile")
        for k, v in (("name", DEMO_NAME), ("age", "30"), ("weight", "62"), ("height", "165"),
                     ("waist", "72"), ("hip", "95"), ("location", "Thane")):
            app.v[k].set(v)
        app.v["job"].set("standing")
        app._submit_profile()
        app.ask_var.set("my lower back hurts and I only have 10 minutes")
        app._ask()
        shot("2-routine")
        app.show_gallery()
        for k in ("tadasana", "vrikshasana", "utkatasana"):
            app.pick_vars[k].set(True)
            app.picked.append(k)
        app.show_gallery()
        shot("3-library")
        app._build_custom()
        app.voice.set(False)
        shot("4-plan")
        app.start_practice()
        shot("5-practice")
        app.results = [
            {"pose": "Mountain Pose", "result": "held 20.0 s, best alignment 91%", "completed": True,
             "scored": True},
            {"pose": "Tree Pose", "result": "held 20.0 s, best alignment 96%", "completed": True,
             "scored": True},
            {"pose": "Chair Pose", "result": "not completed", "completed": False, "scored": True}]
        app._saved_path = "data/users/" + DEMO_NAME + "_history.json"     # not this machine's temp path
        app.show_summary()
        shot("6-summary")
    finally:
        app.destroy()
    return written


def hud_shots() -> list[str]:
    import cv2
    import numpy as np

    from tools.evaluation import pose_check as pc
    from yoga.landmarks import PoseTracker

    tmp = tempfile.mkdtemp()
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    written: list[str] = []
    try:
        for key in HUD_POSES:
            pc.check_pose(tracker, key, tmp)
            dst = os.path.join(OUT, f"hud-{key}.png")
            shutil.copyfile(os.path.join(tmp, key + ".png"), dst)
            written.append(dst)
        # the pose list and the large demo, drawn over one of those camera views
        import app as trainer
        from yoga import demo, overlay, routines
        base = cv2.imread(os.path.join(tmp, "utkatasana.png"))
        menu = base.copy()
        overlay.draw_pose_menu(menu, trainer.pose_menu_entries(), "utkatasana", hover="pose:vrikshasana")
        dst = os.path.join(OUT, "hud-pose-menu.png")
        cv2.imwrite(dst, menu)
        written.append(dst)
        big = base.copy()
        clip = demo.clip("utkatasana")
        overlay.draw_demo_panel(big, clip, 4.5, routines.POSES["utkatasana"].steps[2], "DEMO  UTKATASANA")
        dst = os.path.join(OUT, "hud-demo.png")
        cv2.imwrite(dst, big)
        written.append(dst)
    finally:
        tracker.close()
        shutil.rmtree(tmp, ignore_errors=True)
        cv2.destroyAllWindows()
    return written


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for path in launcher_shots() + hud_shots():
        print("  wrote", os.path.relpath(path, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
