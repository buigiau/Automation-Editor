"""Simple runtime picker for Premiere, Cubase, source video, and audio folder."""

from __future__ import annotations

import threading
from copy import deepcopy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from autoedit.config import apply_inputs, load_config, source_paths
from autoedit.validate import validate_inputs


def _browse_file(var: tk.StringVar, title: str, patterns: list[tuple[str, str]]) -> None:
    path = filedialog.askopenfilename(title=title, filetypes=patterns + [("All files", "*.*")])
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def launch(config_path: str | None = None, legacy: bool = False) -> int:
    if not legacy:
        from autoedit.ai.gui import launch as launch_ai
        return launch_ai(config_path)
    # The ordinary GUI command should use the visible configuration file too.
    default_config = Path(__file__).resolve().parents[3] / "config.yaml"
    cfg = load_config(config_path or (default_config if default_config.is_file() else None))
    root = tk.Tk()
    root.title("AutoEdit")
    root.minsize(720, 420)

    premiere_var = tk.StringVar(value=(cfg.get("premiere") or {}).get("project") or "")
    cubase_var = tk.StringVar(value=(cfg.get("cubase") or {}).get("project") or "")
    videos = source_paths((cfg.get("premiere") or {}).get("source_media"))
    audio_var = tk.StringVar(value=(cfg.get("audio") or {}).get("directory") or "")
    output_var = tk.StringVar(value=(cfg.get("job") or {}).get("output_dir") or str(Path("output").resolve()))
    kind_var = tk.StringVar(value=(cfg.get("video") or {}).get("source_kind") or "live_action")
    gap_var = tk.StringVar(value=str((cfg.get("video") or {}).get("source_gap_sec", 5.0)))
    lip_var = tk.BooleanVar(value=(cfg.get("video") or {}).get("require_lip_motion", False))
    clear_face_var = tk.BooleanVar(value=(cfg.get('video') or {}).get('require_clear_face', True))
    fast_review_var = tk.BooleanVar(value=(cfg.get('video') or {}).get('verification_mode') == 'review')
    analysis_labels = {'Stop when all slots are filled': 'until_filled', 'Analyze all sources': 'full'}
    analysis_var = tk.StringVar(value=next(label for label, mode in analysis_labels.items()
        if mode == (cfg.get('video') or {}).get('analysis_mode', 'until_filled')))
    sampler_selection = (cfg.get("cubase") or {}).get("sampler_tracks") or ""
    sampler_var = tk.StringVar(value=", ".join(map(str, sampler_selection))
                              if isinstance(sampler_selection, list) else sampler_selection)
    references = list(((cfg.get('video') or {}).get('characters') or {}).get('reference_images', []))
    reference_regions = deepcopy(((cfg.get('video') or {}).get('characters') or {}).get('reference_regions', {}))
    cancel_event = threading.Event()
    active = threading.Event()
    reference_var = tk.StringVar(value=f'{len(references)} images' if references else 'Automatic character selection')
    def import_references():
        paths = filedialog.askopenfilenames(title='Select character portraits or group images',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp')])
        if paths:
            references[:] = paths
            review_references()
    def review_references():
        from autoedit.reference_picker import select_regions
        for path in references:
            key = str(Path(path).resolve())
            try:
                regions = select_regions(root,path,kind_var.get(),reference_regions.get(key))
            except Exception as exc:
                messagebox.showerror('Target characters',str(exc))
                return False
            if regions is None:
                return False
            reference_regions[key] = regions
        reference_var.set(f'{len(references)} images; {sum(len(reference_regions.get(str(Path(p).resolve()), [])) for p in references)} selected targets'
                          if references else 'Automatic character selection')
        return True
    def clear_references():
        references.clear()
        reference_regions.clear()
        reference_var.set('Automatic character selection')

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
        ("Source Videos", None, None),
        ("Audio Folder", audio_var, lambda: _browse_dir(audio_var, "Select audio folder")),
        ("Output Folder", output_var, lambda: _browse_dir(output_var, "Select output folder")),
        ('Target characters (optional)', reference_var, import_references),
    ]
    for i, (label, var, browse) in enumerate(rows):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 8))
        if var is None:
            video_list = tk.Listbox(frm, height=4, selectmode=tk.EXTENDED, exportselection=False)
            video_list.grid(row=i, column=1, sticky="ew", pady=4)
            for path in videos:
                video_list.insert(tk.END, path)

            def add_videos():
                selected = filedialog.askopenfilenames(
                    title="Add source videos",
                    filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mpg *.mpeg *.mxf"),
                               ("All files", "*.*")])
                if selected:
                    paths = source_paths(list(video_list.get(0, tk.END)) + list(selected))
                    video_list.delete(0, tk.END)
                    for path in paths:
                        video_list.insert(tk.END, path)

            def remove_videos():
                for index in reversed(video_list.curselection()):
                    video_list.delete(index)

            video_buttons = ttk.Frame(frm)
            video_buttons.grid(row=i, column=2, padx=(8, 0))
            ttk.Button(video_buttons, text="Add videos…", command=add_videos).pack(fill=tk.X)
            ttk.Button(video_buttons, text="Remove selected", command=remove_videos).pack(fill=tk.X, pady=(4, 0))
            continue
        ttk.Entry(frm, textvariable=var, state='readonly' if var is reference_var else 'normal').grid(row=i, column=1, sticky="ew", pady=4)
        if var is reference_var:
            buttons = ttk.Frame(frm)
            buttons.grid(row=i, column=2, padx=(8, 0))
            ttk.Button(buttons, text='Import images', command=import_references).pack(side=tk.LEFT)
            ttk.Button(buttons, text='Review / crop', command=review_references).pack(side=tk.LEFT)
            ttk.Button(buttons, text='Clear', command=clear_references).pack(side=tk.LEFT)
            continue
        ttk.Button(frm, text="Browse…", command=browse).grid(row=i, column=2, padx=(8, 0), pady=4)

    log = tk.Text(frm, height=12, wrap="word")
    ttk.Label(frm, text="Source Type").grid(row=len(rows), column=0, sticky="w")
    ttk.Combobox(frm, textvariable=kind_var, values=("live_action", "animation"), state="readonly").grid(row=len(rows), column=1, sticky="ew")
    ttk.Label(frm, text="Source gap (seconds)").grid(row=len(rows) + 1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=gap_var).grid(row=len(rows) + 1, column=1, sticky="ew")
    ttk.Label(frm, text="Sampler tracks").grid(row=len(rows) + 2, column=0, sticky="w")
    ttk.Entry(frm, textvariable=sampler_var).grid(row=len(rows) + 2, column=1, sticky="ew")
    ttk.Label(frm, text="Blank = all; e.g. 2-13").grid(row=len(rows) + 2, column=2, padx=(8, 0))
    ttk.Checkbutton(frm, text="Require speaking mouths",
                    variable=lip_var).grid(row=len(rows) + 3, column=1, columnspan=2, sticky="w")
    ttk.Label(frm, text='Analysis').grid(row=len(rows) + 4, column=0, sticky='w')
    ttk.Combobox(frm, textvariable=analysis_var, values=list(analysis_labels), state='readonly').grid(
        row=len(rows) + 4, column=1, sticky='ew')
    ttk.Label(frm, text='Ranks analyzed character exposure').grid(row=len(rows) + 4, column=2, padx=(8,0))
    ttk.Checkbutton(frm, text='Require clear faces (off allows body shots)', variable=clear_face_var).grid(
        row=len(rows) + 5, column=1, columnspan=2, sticky='w')
    ttk.Checkbutton(frm, text='Fast export; review faces, lips and voice afterward',
                    variable=fast_review_var).grid(row=len(rows) + 6, column=1, columnspan=2, sticky='w')
    log.grid(row=len(rows) + 7, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
    frm.rowconfigure(len(rows) + 7, weight=1)

    def write(msg: str) -> None:
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        root.update_idletasks()

    def current_kwargs() -> dict:
        return {
            "premiere_project": premiere_var.get().strip(),
            "cubase_project": cubase_var.get().strip(),
            "source_video": list(video_list.get(0, tk.END)),
            "audio_directory": audio_var.get().strip(),
            "template_sequence": (cfg.get("premiere") or {}).get("template_sequence") or "PJ 5 - demo",
            "video_track_index": int((cfg.get("premiere") or {}).get("video_track_index", 1)),
            "sampler_tracks": sampler_var.get().strip(),
        }

    def do_validate() -> list[str]:
        errors = validate_inputs(**current_kwargs())
        for path in references:
            if not Path(path).is_file():
                errors.append(f'Character reference missing: {path}')
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
            for path in video_list.get(0, tk.END):
                write("Video:    " + path)
            write("Audio:    " + audio_var.get().strip())
        return errors

    def do_run() -> None:
        errors = do_validate()
        if errors:
            messagebox.showerror("AutoEdit", "Fix the selected inputs before running.")
            return
        if references and any(str(Path(p).resolve()) not in reference_regions for p in references):
            if not review_references():
                return
        cancel_event.clear()
        active.set()
        cancel_btn.state(['!disabled'])
        run_btn.state(["disabled"])
        validate_btn.state(["disabled"])
        import_btn.state(["disabled"])
        apply_inputs(cfg, premiere=premiere_var.get().strip(), cubase=cubase_var.get().strip(),
                     video=list(video_list.get(0, tk.END)), audio_dir=audio_var.get().strip(),
                     output_dir=output_var.get().strip() or None)
        cfg.setdefault("video", {}).update(source_kind=kind_var.get(), source_gap_sec=float(gap_var.get()),
                                           require_lip_motion=lip_var.get(), require_clear_face=clear_face_var.get(),
                                           analysis_mode=analysis_labels[analysis_var.get()])
        cfg['video']['verification_mode'] = 'review' if fast_review_var.get() else 'verified'
        cfg["video"].setdefault("characters", {})["main_group"] = "auto"
        cfg['video']['characters']['reference_images'] = list(references)
        cfg['video']['characters']['reference_regions'] = deepcopy(reference_regions)
        cfg.setdefault("cubase", {})["sampler_tracks"] = sampler_var.get().strip()
        job_config = deepcopy(cfg)

        def worker() -> None:
            from autoedit.worker import run_fresh

            try:
                result = run_fresh(job_config, log=lambda m: root.after(0, write, m), cancel_event=cancel_event)
                plan = result["plan"]
                unmatched = [
                    s for s in plan.get("slots") or [] if s.get("match_status") in ("unmatched", "empty")
                ]
                root.after(0, write, f"edit-plan: {result['plan_path']}")
                root.after(0, write, f"slots={len(plan.get('slots') or [])} unmatched={len(unmatched)}")
                if plan.get('review_status') == 'manual_review_required':
                    root.after(0, write, 'Fast export ready. Review every scene and voice; dense face/lip/onset checks were deferred.')
                if plan.get("audio_mode") == "sampler_per_scene_v1":
                    track_names = ", ".join(s["cubase"]["track_name"] for s in plan["cubase_slots"])
                    root.after(0, write, f"Cubase: {len(plan['cubase_slots'])} samples ready. Open the paired project in Cubase 13, manually select exactly these tracks: {track_names}, then click Import Cubase. Samples load into the open project; the tool verifies but does not save it.")
                    root.after(0, write, "Export the Cubase mix to cubase_mixdown.wav, then use Import mixdown WAV in Premiere. The preview is not a Cubase MIDI/FX render.")
                root.after(
                    0,
                    write,
                    "Open this project in Premiere, load the edit-plan in AutoEdit, then Apply. "
                    "The panel will not save the project.",
                )
            except Exception as exc:
                from autoedit.runtime import JobCancelled
                root.after(0, write, f"error: {exc}")
                if not isinstance(exc, JobCancelled):
                    root.after(0, lambda message=str(exc): messagebox.showerror("AutoEdit", message))
            finally:
                active.clear()
                root.after(0, lambda: cancel_btn.state(['disabled']))
                root.after(0, lambda: run_btn.state(["!disabled"]))
                root.after(0, lambda: validate_btn.state(["!disabled"]))
                root.after(0, lambda: import_btn.state(["!disabled"]))

        threading.Thread(target=worker, daemon=True).start()

    btns = ttk.Frame(frm)
    btns.grid(row=len(rows) + 8, column=0, columnspan=3, sticky="w", pady=8)
    validate_btn = ttk.Button(btns, text="Validate", command=do_validate)
    validate_btn.pack(side=tk.LEFT, padx=(0, 8))
    run_btn = ttk.Button(btns, text="Run", command=do_run)
    run_btn.pack(side=tk.LEFT)
    cancel_btn = ttk.Button(btns, text='Cancel', command=cancel_event.set, state='disabled')
    cancel_btn.pack(side=tk.LEFT, padx=(8, 0))

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

    write("Select Premiere .prproj, Cubase .cpr, one or more source videos, and audio folder, then Validate / Run.")
    write("Import Cubase uses the tracks you select manually in Cubase; it does not change the track selection.")
    write('Jobs from other AutoEdit windows queue automatically. Cancel preserves completed analysis for the next run.')
    def close_window():
        if active.is_set():
            cancel_event.set()
            root.after(200, close_window)
        else:
            root.destroy()
    root.protocol('WM_DELETE_WINDOW', close_window)
    root.mainloop()
    return 0
