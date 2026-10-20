"""Preset-first Tk UI. All worker messages are delivered through a main-thread queue."""
from copy import deepcopy
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser

from autoedit.config import load_config, source_paths
from autoedit.presets import PRESETS, ROOT, load_settings, save_settings, apply_preset


class Application:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.settings = load_settings()
        self.references, self.thumbnails = [], []
        self.events, self.cancel = queue.Queue(), threading.Event()
        self.active = False
        self.closing = False
        root.title('AutoEdit — Preset + AI')
        root.minsize(780, 650)
        self.preset = tk.StringVar(value=PRESETS.get(self.settings['last_preset'], 'Soda Pop'))
        self.output = tk.StringVar(value=self.settings['last_output'])
        self.status = tk.StringVar(value='Chọn preset, output và video nguồn rồi bấm Chạy.')
        frm = ttk.Frame(root, padding=16)
        frm.pack(fill='both', expand=True)
        frm.columnconfigure(1, weight=1)
        ttk.Label(frm, text='Preset').grid(row=0, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.preset, values=list(PRESETS.values()), state='readonly').grid(row=0, column=1, sticky='ew')
        self.settings_button = ttk.Button(frm, text='Thiết lập', command=self.edit_settings)
        self.settings_button.grid(row=0, column=2, padx=8)
        ttk.Label(frm, text='Folder output').grid(row=1, column=0, sticky='w', pady=8)
        ttk.Entry(frm, textvariable=self.output).grid(row=1, column=1, sticky='ew')
        ttk.Button(frm, text='Chọn folder', command=lambda: self.browse_dir(self.output)).grid(row=1, column=2)
        ttk.Label(frm, text='Video nguồn').grid(row=2, column=0, sticky='nw')
        self.videos = tk.Listbox(frm, height=4, selectmode='extended', exportselection=False)
        self.videos.grid(row=2, column=1, sticky='ew')
        buttons = ttk.Frame(frm)
        buttons.grid(row=2, column=2, padx=8)
        ttk.Button(buttons, text='Thêm video', command=self.add_videos).pack(fill='x')
        ttk.Button(buttons, text='Xóa đã chọn', command=self.remove_videos).pack(fill='x', pady=4)
        refs = ttk.LabelFrame(frm, text='Ảnh nhân vật — tùy chọn', padding=8)
        refs.grid(row=3, column=0, columnspan=3, sticky='ew', pady=12)
        ttk.Button(refs, text='Thêm ảnh', command=self.add_references).pack(anchor='w')
        self.reference_frame = ttk.Frame(refs)
        self.reference_frame.pack(fill='x')
        self.refresh_references()
        controls = ttk.Frame(frm)
        controls.grid(row=4, column=0, columnspan=3, sticky='w')
        self.run_button = ttk.Button(controls, text='Chạy', command=self.run)
        self.run_button.pack(side='left')
        self.cancel_button = ttk.Button(controls, text='Hủy', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=8)
        ttk.Button(controls, text='Xem / nghe review', command=self.open_review).pack(side='left')
        self.edits_button = ttk.Button(controls, text='Áp dụng chỉnh sửa', command=self.apply_edits)
        self.edits_button.pack(side='left', padx=8)
        self.import_button = ttk.Button(controls, text='Import Cubase', command=self.import_cubase)
        self.import_button.pack(side='left')
        self.progress = ttk.Progressbar(frm, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky='ew', pady=10)
        ttk.Label(frm, textvariable=self.status, wraplength=720).grid(row=6, column=0, columnspan=3, sticky='w')
        self.log = tk.Text(frm, height=12, wrap='word', state='disabled')
        self.log.grid(row=7, column=0, columnspan=3, sticky='nsew', pady=8)
        frm.rowconfigure(7, weight=1)
        ttk.Label(frm, text='Apply Premiere: mở panel AutoEdit trong Premiere và tải edit-plan.json.', wraplength=720).grid(row=8, column=0, columnspan=3, sticky='w')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll)

    @staticmethod
    def browse_dir(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def add_videos(self):
        paths = filedialog.askopenfilenames(title='Video nguồn', filetypes=[('Video', '*.mp4 *.mov *.mkv *.avi *.m4v *.mxf *.wmv *.mpg *.mpeg')])
        for path in source_paths(list(self.videos.get(0, 'end')) + list(paths)):
            if path not in self.videos.get(0, 'end'):
                self.videos.insert('end', path)

    def remove_videos(self):
        for index in reversed(self.videos.curselection()):
            self.videos.delete(index)

    def add_references(self):
        from PIL import Image
        paths = filedialog.askopenfilenames(title='Ảnh nhân vật / nhiều góc', filetypes=[('Ảnh', '*.png *.jpg *.jpeg *.webp *.bmp')])
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                messagebox.showerror('Ảnh không đọc được', str(exc))
                continue
            if path not in self.references:
                self.references.append(path)
        self.refresh_references()

    def refresh_references(self):
        from PIL import Image, ImageTk, ImageOps
        for child in self.reference_frame.winfo_children():
            child.destroy()
        self.thumbnails.clear()
        if not self.references:
            ttk.Label(self.reference_frame, text='Tự chọn nhân vật theo thời lượng xuất hiện.').pack(anchor='w', pady=8)
        for index, path in enumerate(self.references):
            card = ttk.Frame(self.reference_frame)
            card.grid(row=index // 6, column=index % 6, padx=8, pady=6)
            with Image.open(path) as img:
                photo = ImageTk.PhotoImage(ImageOps.contain(img.convert('RGB'), (84, 72)))
            self.thumbnails.append(photo)
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=Path(path).name[:18]).pack()
            def remove(p=path):
                self.references.remove(p)
                self.refresh_references()
            ttk.Button(card, text='Xóa', command=remove).pack()

    def edit_settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title('Thiết lập một lần')
        dialog.transient(self.root)
        dialog.grab_set()
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill='both', expand=True, padx=12, pady=12)
        general, advanced = ttk.Frame(notebook, padding=12), ttk.Frame(notebook, padding=12)
        notebook.add(general, text='Thiết lập')
        notebook.add(advanced, text='Nâng cao')
        fields, pairs = {}, {}
        def entry(parent, row, label, value, browse=None, secret=False):
            var = tk.StringVar(value=value)
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=4, pady=4)
            ttk.Entry(parent, textvariable=var, width=60, show='*' if secret else '').grid(row=row, column=1, sticky='ew')
            if browse:
                def pick():
                    path = (filedialog.askdirectory(parent=dialog) if browse == 'dir' else
                            filedialog.askopenfilename(parent=dialog, filetypes=[('Project', browse)]))
                    if path:
                        var.set(path)
                ttk.Button(parent, text='Chọn', command=pick).grid(row=row, column=2)
            return var
        for i, (key, label) in enumerate(PRESETS.items()):
            pair = self.settings['presets'][key]
            pairs[key] = {
                'premiere_project': entry(general, i*2, label+' · Premiere', pair['premiere_project'], '*.prproj'),
                'cubase_project': entry(general, i*2+1, label+' · Cubase', pair['cubase_project'], '*.cpr'),
                'template_sequence': entry(advanced, i*3, label+' · Sequence', pair['template_sequence']),
                'video_track_index': entry(advanced, i*3+1, label+' · Video track (0-based)', pair['video_track_index']),
                'sampler_tracks': entry(advanced, i*3+2, label+' · Sampler tracks', pair['sampler_tracks'])}
        fields['voice_directory'] = entry(general, 6, 'Thư viện Voice', self.settings['voice_directory'], 'dir')
        fields['api_key'] = entry(general, 7, 'Gemini API key', self.settings['api_key'], secret=True)
        fields['backend'] = tk.StringVar(value=self.settings['backend'])
        ttk.Label(general, text='AI backend').grid(row=8, column=0, sticky='w', padx=4, pady=4)
        ttk.Combobox(general, textvariable=fields['backend'], values=['local', 'gemini'], state='readonly').grid(row=8, column=1, sticky='ew')
        fields['local_model'] = entry(general, 9, 'Model local', self.settings['local_model'])
        fields['local_url'] = entry(general, 10, 'Ollama local URL', self.settings['local_url'])
        ttk.Label(general, text='Local: không cần key, video xử lý trên máy; Voice cần nghe kiểm tra.\nGemini: gửi video gốc có tiếng lên Google; dùng key đã lưu trên máy.').grid(row=11, column=0, columnspan=3, pady=8)
        def setup():
            model = fields['local_model'].get().strip()
            if model not in ('qwen3-vl:4b-instruct', 'qwen3-vl:2b-instruct'):
                messagebox.showerror('Qwen local', 'Trình cài hỗ trợ qwen3-vl:4b-instruct hoặc qwen3-vl:2b-instruct.', parent=dialog)
                return
            dialog.destroy()
            self.setup_local(model)
        ttk.Button(general, text='Cài / kiểm tra Qwen local', command=setup).grid(row=12, column=0, columnspan=3)
        fields['local_device'] = tk.StringVar(value=self.settings['local_device'])
        ttk.Label(general, text='Tài nguyên local (CPU mặc định)').grid(row=13, column=0, sticky='w', padx=4, pady=4)
        ttk.Combobox(general, textvariable=fields['local_device'], values=['cpu', 'gpu'], state='readonly').grid(row=13, column=1, sticky='ew')
        ttk.Label(general, text='CPU chậm hơn, giữ GPU cho màn hình. Chỉ chọn GPU khi máy rảnh.').grid(row=14, column=0, columnspan=3, pady=4)
        fields['model'] = entry(advanced, 9, 'Model Gemini', self.settings['model'])
        fields['source_gap_sec'] = entry(advanced, 10, 'Khoảng cách nguồn (giây)', self.settings['source_gap_sec'])
        fields['cache_dir'] = entry(advanced, 11, 'Cache dùng chung', self.settings['cache_dir'], 'dir')
        fields['video_fps'] = entry(advanced, 12, 'Khung hình phân tích / giây', self.settings['video_fps'])
        fields['max_source_seconds'] = entry(advanced, 13, 'Giới hạn giây / nguồn (0 = toàn bộ)', self.settings['max_source_seconds'])
        fallback_var = tk.BooleanVar(value=self.settings['shot_fallback'])
        ttk.Checkbutton(advanced,text='Thiếu shot: cho phép đoạn khác trong cùng shot',variable=fallback_var).grid(row=14,column=0,columnspan=3,sticky='w')
        ttk.Label(advanced, text='Sampler: để trống = tất cả; hoặc 2-13 / 1,3-5.').grid(row=15, column=0, columnspan=3)
        fields['local_batch_frames'] = entry(advanced, 16, 'Local: số khung hình / lô (6–12)', self.settings['local_batch_frames'])
        fields['local_threads'] = entry(advanced, 17, 'Local: số luồng CPU (1–8)', self.settings['local_threads'])
        fields['local_gpu_layers'] = entry(advanced, 18, 'Local: lớp trên GPU (chỉ dùng khi chọn GPU)', self.settings['local_gpu_layers'])
        fields['local_analysis_mode'] = tk.StringVar(value=self.settings['local_analysis_mode'])
        ttk.Label(advanced, text='Local: phạm vi tìm cảnh').grid(row=19, column=0, sticky='w')
        ttk.Combobox(advanced, textvariable=fields['local_analysis_mode'], values=['until_filled', 'full'], state='readonly').grid(row=19, column=1, sticky='ew')
        fields['local_target_seconds'] = entry(advanced, 20, 'Local: thời gian mục tiêu (giây)', self.settings['local_target_seconds'])
        fields['local_max_seconds'] = entry(advanced, 21, 'Local: giới hạn cứng (0 = tìm đến đủ)', self.settings['local_max_seconds'])
        fields['transition_tail_fallback_sec'] = entry(advanced, 22, 'Các cục trước: chuyển cảnh đuôi (0–2 giây)', self.settings['transition_tail_fallback_sec'])
        final_fallback_var = tk.BooleanVar(value=self.settings['final_scene_transition_fallback'])
        ttk.Checkbutton(advanced, text='Cục cuối timeline: cho phép chuyển cảnh không giới hạn',
            variable=final_fallback_var).grid(row=23, column=0, columnspan=3, sticky='w')
        fields['shortage_policy'] = tk.StringVar(value=self.settings.get('shortage_policy', 'complete'))
        ttk.Label(advanced, text='Thiếu cảnh: complete = xuất bản nháp; strict = dừng').grid(row=24, column=0, sticky='w')
        ttk.Combobox(advanced, textvariable=fields['shortage_policy'], values=['complete', 'strict'], state='readonly').grid(row=24, column=1, sticky='ew')
        def save():
            import math
            try:
                settings = deepcopy(self.settings)
                for key, values in pairs.items():
                    settings['presets'][key] = {k: v.get().strip() for k, v in values.items()}
                    settings['presets'][key]['video_track_index'] = int(values['video_track_index'].get())
                    if settings['presets'][key]['video_track_index'] < 0:
                        raise ValueError('Video track phải >= 0')
                settings.update({k: v.get().strip() for k, v in fields.items()})
                from autoedit.ai.sampling import sampling_settings
                settings.update(sampling_settings({**settings,'shot_fallback':fallback_var.get(),
                    'final_scene_transition_fallback': final_fallback_var.get()}))
                from autoedit.ai.local import local_settings
                from autoedit.ai.ollama import validate_endpoint, resource_settings
                settings['local_batch_frames'] = int(settings['local_batch_frames'])
                settings['local_threads'] = int(settings['local_threads'])
                settings['local_gpu_layers'] = int(settings['local_gpu_layers'])
                settings['local_target_seconds'] = float(settings['local_target_seconds'])
                settings['local_max_seconds'] = float(settings['local_max_seconds'])
                local_settings(settings)
                from autoedit.ai.completion import shortage_policy
                shortage_policy(settings)
                resource_settings(settings)
                validate_endpoint(settings['local_url'])
                settings['source_gap_sec'] = float(settings['source_gap_sec'])
                if not math.isfinite(settings['source_gap_sec']) or settings['source_gap_sec'] < 0:
                    raise ValueError('Khoảng cách nguồn phải >= 0')
                save_settings(settings)
                self.settings = settings
                dialog.destroy()
            except (ValueError, OSError) as exc:
                messagebox.showerror('Thiết lập', str(exc), parent=dialog)
        ttk.Button(dialog, text='Lưu', command=save).pack(pady=12)

    def setup_local(self, model=None):
        model = model or self.settings['local_model']
        def work():
            import subprocess
            import sys
            from autoedit.worker import _stop_process_tree
            executable = Path(sys.executable).with_name('python.exe') if sys.executable.lower().endswith('pythonw.exe') else Path(sys.executable)
            done = threading.Event()
            with subprocess.Popen([str(executable), '-u', str(ROOT / 'scripts/setup_local_ai.py'), '--activate', '--model', model],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                def watch():
                    while not done.wait(.2):
                        if self.cancel.is_set():
                            _stop_process_tree(process)
                            return
                watcher = threading.Thread(target=watch, daemon=True)
                watcher.start()
                try:
                    for line in process.stdout:
                        self.events.put(('log', line.rstrip()))
                    if process.wait():
                        raise RuntimeError('Cài Qwen chưa hoàn tất; xem log phía trên và chạy lại để tiếp tục.')
                    self.events.put(('reload_settings', None))
                finally:
                    done.set()
                    watcher.join(timeout=2)
        self.start(work)

    def append_log(self, text):
        self.status.set(text)
        self.log.configure(state='normal')
        self.log.insert('end', text+'\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def start(self, operation):
        if self.active:
            return
        self.active = True
        self.cancel.clear()
        for button in (self.run_button, self.edits_button, self.import_button, self.settings_button):
            button.state(['disabled'])
        self.cancel_button.state(['!disabled'])
        self.progress.start(15)
        def work():
            try:
                operation()
                self.events.put(('done', 'Hoàn tất.'))
            except Exception as exc:
                self.events.put(('error', str(exc)))
            finally:
                self.events.put(('idle', None))
        threading.Thread(target=work, daemon=True).start()

    def run(self):
        key = next(k for k, label in PRESETS.items() if label == self.preset.get())
        try:
            cfg = apply_preset(self.config, self.settings, key)
            cfg['premiere']['source_media'] = list(self.videos.get(0, 'end'))
            cfg['job']['output_dir'] = str(Path(self.output.get().strip() or 'output').resolve())
            cfg.setdefault('video', {}).setdefault('characters', {})['reference_images'] = list(self.references)
            cfg['video']['characters']['reference_regions'] = {}
            self.settings.update(last_preset=key, last_output=cfg['job']['output_dir'])
            save_settings(self.settings)
        except (ValueError, OSError) as exc:
            messagebox.showerror('AutoEdit', str(exc))
            return
        def work():
            from autoedit.worker import run_fresh
            result = run_fresh(cfg, log=lambda m: self.events.put(('log', m)), cancel_event=self.cancel)
            self.events.put(('log', 'Bộ autofill: '+result['plan_path']))
        self.start(work)

    def open_review(self):
        path = Path(self.output.get()) / 'ai_review.html'
        if path.is_file():
            webbrowser.open(path.resolve().as_uri())
        else:
            messagebox.showinfo('Review', 'Chạy trước để tạo trang review.')

    def apply_edits(self):
        path = filedialog.askopenfilename(title='Chọn autoedit-corrections.json', filetypes=[('JSON', '*.json')])
        if not path:
            return
        cfg = {'job': {'output_dir': self.output.get()}, '_ai_review': path}
        def work():
            from autoedit.worker import run_fresh
            run_fresh(cfg, log=lambda m: self.events.put(('log', m)), cancel_event=self.cancel)
        self.start(work)

    def import_cubase(self):
        path = Path(self.output.get()) / 'edit-plan.json'
        if not path.is_file():
            messagebox.showerror('Cubase', 'Chưa có bộ autofill hoàn chỉnh.')
            return
        def work():
            from autoedit.cubase.host import import_samples
            import_samples(path, log=lambda m: self.events.put(('log', m)))
        self.start(work)
        # Existing Cubase importer controls its own host transaction; no process to kill here.
        self.cancel_button.state(['disabled'])

    def poll(self):
        while not self.events.empty():
            kind, text = self.events.get_nowait()
            if kind == 'reload_settings':
                self.settings = load_settings()
            elif kind == 'idle':
                self.active = False
                self.progress.stop()
                self.cancel_button.state(['disabled'])
                for button in (self.run_button, self.edits_button, self.import_button, self.settings_button):
                    button.state(['!disabled'])
            else:
                self.append_log(text)
        if self.closing and not self.active:
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def close(self):
        self.closing = True
        self.cancel.set()


def launch(config_path=None):
    default = ROOT / 'config.yaml'
    cfg = load_config(config_path or (default if default.is_file() else None))
    root = tk.Tk()
    Application(root, cfg)
    root.mainloop()
    return 0
