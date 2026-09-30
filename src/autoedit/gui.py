"""Simple runtime picker for Premiere, Cubase, source video, and audio folder."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from autoedit.config import apply_inputs, load_config
from autoedit.validate import validate_inputs


def _browse_file(var: tk.StringVar, title: str, patterns: list[tuple[str, str]]) -> None:
    path = filedialog.askopenfilename(title=title, filetypes=patterns + [("All files", "*.*")])
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def launch(config_path: str | None = None) -> int:
    # The ordinary GUI command should use the visible configuration file too.
    default_config = Path(__file__).resolve().parents[3] / "config.yaml"
    cfg = load_config(config_path or (default_config if default_config.is_file() else None))
    root = tk.Tk()
    root.title("AutoEdit")
    root.minsize(720, 420)

    premiere_var = tk.StringVar(value=(cfg.get("premiere") or {}).get("project") or "")
    cubase_var = tk.StringVar(value=(cfg.get("cubase") or {}).get("project") or "")
    video_var = tk.StringVar(value=(cfg.get("premiere") or {}).get("source_media") or "")
    audio_var = tk.StringVar(value=(cfg.get("audio") or {}).get("directory") or "")
    output_var = tk.StringVar(value=(cfg.get("job") or {}).get("output_dir") or str(Path("output").resolve()))
    kind_var = tk.StringVar(value=(cfg.get("video") or {}).get("source_kind") or "live_action")
    gap_var = tk.StringVar(value=str((cfg.get("video") or {}).get("source_gap_sec", 5.0)))

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill=tk.BOTH, expand=True)
    frm.columnconfigure(1, weight=1)

    rows = [
        ("Premiere Project", premiere_var, lambda: _browse_file(
            premiere_var, "Select Premiere project", [("Premiere project", "*.prproj")]
        )),
        ("Cubase Project", cubase_var, lambda: _browse_file(
            cubase_var, "Select Cubase project", [("Cubase project", "*.cpr")]
        )),
        ("Source Video", video_var, lambda: _browse_file(
            video_var,
            "Select source video",
            [("Video", "*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mxf"), ("MP4", "*.mp4")],
        )),
        ("Audio Folder", audio_var, lambda: _browse_dir(audio_var, "Select audio folder")),
        ("Output Folder", output_var, lambda: _browse_dir(output_var, "Select output folder")),
    ]
    for i, (label, var, browse) in enumerate(rows):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Entry(frm, textvariable=var).grid(row=i, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Browse…", command=browse).grid(row=i, column=2, padx=(8, 0), pady=4)

    log = tk.Text(frm, height=12, wrap="word")
    ttk.Label(frm, text="Source Type").grid(row=len(rows), column=0, sticky="w")
    ttk.Combobox(frm, textvariable=kind_var, values=("live_action", "animation"), state="readonly").grid(row=len(rows), column=1, sticky="ew")
    ttk.Label(frm, text="Source gap (seconds)").grid(row=len(rows) + 1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=gap_var).grid(row=len(rows) + 1, column=1, sticky="ew")
    log.grid(row=len(rows) + 3, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
    frm.rowconfigure(len(rows) + 3, weight=1)

    def write(msg: str) -> None:
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        root.update_idletasks()

    def current_kwargs() -> dict:
        return {
            "premiere_project": premiere_var.get().strip(),
            "cubase_project": cubase_var.get().strip(),
            "source_video": video_var.get().strip(),
            "audio_directory": audio_var.get().strip(),
        }

    def do_validate() -> list[str]:
        errors = validate_inputs(**current_kwargs())
        try:
            import math
            gap = float(gap_var.get())
            if not math.isfinite(gap) or gap < 0:
                raise ValueError()
        except ValueError:
            errors.append("Source gap must be a non-negative number of seconds.")
        log.delete("1.0", tk.END)
        if errors:
            write("Validation failed:")
            for err in errors:
                write("  - " + err)
        else:
            write("Inputs look valid.")
            write("Premiere: " + premiere_var.get().strip())
            write("Cubase:   " + cubase_var.get().strip())
            write("Video:    " + video_var.get().strip())
            write("Audio:    " + audio_var.get().strip())
        return errors

    def do_run() -> None:
        errors = do_validate()
        if errors:
            messagebox.showerror("AutoEdit", "Fix the selected inputs before running.")
            return
        run_btn.state(["disabled"])
        validate_btn.state(["disabled"])
        import_btn.state(["disabled"])
        apply_inputs(cfg, premiere=premiere_var.get().strip(), cubase=cubase_var.get().strip(),
                     video=video_var.get().strip(), audio_dir=audio_var.get().strip(),
                     output_dir=output_var.get().strip() or None)
        cfg.setdefault("video", {}).update(source_kind=kind_var.get(), source_gap_sec=float(gap_var.get()))

        def worker() -> None:
            from autoedit.worker import run_fresh

            try:
                result = run_fresh(cfg, log=lambda m: root.after(0, write, m))
                plan = result["plan"]
                unmatched = [
                    s for s in plan.get("slots") or [] if s.get("match_status") in ("unmatched", "empty")
                ]
                root.after(0, write, f"edit-plan: {result['plan_path']}")
                root.after(0, write, f"slots={len(plan.get('slots') or [])} unmatched={len(unmatched)}")
                if plan.get("audio_mode") == "sampler_per_scene_v1":
                    root.after(0, write, f"Cubase: {len(plan['cubase_slots'])} samples ready. Open the paired project in Cubase 13, manually select its numbered Sampler Tracks, then click Import Cubase. Samples load into the open project; the tool verifies but does not save it.")
                    root.after(0, write, "Export the Cubase mix to cubase_mixdown.wav, then use Import mixdown WAV in Premiere. The preview is not a Cubase MIDI/FX render.")
                root.after(
                    0,
                    write,
                    "Open this project in Premiere, load the edit-plan in AutoEdit, then Apply. "
                    "The panel will not save the project.",
                )
            except Exception as exc:
                root.after(0, write, f"error: {exc}")
                root.after(0, lambda message=str(exc): messagebox.showerror("AutoEdit", message))
            finally:
                root.after(0, lambda: run_btn.state(["!disabled"]))
                root.after(0, lambda: validate_btn.state(["!disabled"]))
                root.after(0, lambda: import_btn.state(["!disabled"]))

        threading.Thread(target=worker, daemon=True).start()

    btns = ttk.Frame(frm)
    btns.grid(row=len(rows) + 2, column=0, columnspan=3, sticky="w", pady=8)
    validate_btn = ttk.Button(btns, text="Validate", command=do_validate)
    validate_btn.pack(side=tk.LEFT, padx=(0, 8))
    run_btn = ttk.Button(btns, text="Run", command=do_run)
    run_btn.pack(side=tk.LEFT)

    def do_import_cubase():
        plan_path = Path(output_var.get().strip() or "output") / "edit-plan.json"
        if not plan_path.is_file():
            messagebox.showerror("AutoEdit", "Run first to create edit-plan.json.")
            return
        for button in (run_btn, validate_btn, import_btn):
            button.state(["disabled"])
        def worker():
            try:
                from autoedit.cubase.host import import_samples
                import_samples(plan_path, log=lambda m: root.after(0, write, m))
            except Exception as exc:
                root.after(0, write, "Cubase import stopped: " + str(exc))
                root.after(0, lambda msg=str(exc): messagebox.showerror("Cubase import", msg))
            finally:
                for button in (run_btn, validate_btn, import_btn):
                    root.after(0, lambda b=button: b.state(["!disabled"]))
        threading.Thread(target=worker, daemon=True).start()

    import_btn = ttk.Button(btns, text="Import Cubase", command=do_import_cubase)
    import_btn.pack(side=tk.LEFT, padx=(8, 0))

    write("Select Premiere .prproj, Cubase .cpr, source video, and audio folder, then Validate / Run.")
    write("Import Cubase uses the tracks you select manually in Cubase; it does not change the track selection.")
    root.mainloop()
    return 0
