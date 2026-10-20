"""Desktop studio with a sequential job queue and isolated worker processes."""
from copy import deepcopy
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser

from autoedit.config import load_config, source_paths
from autoedit.job_queue import JobQueue
from autoedit.ai import theme
from autoedit.presets import PRESETS, ROOT, load_settings, save_settings, apply_preset


class Application:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.settings = load_settings()
        self.references, self.thumbnails = [], []
        self.events, self.cancel = queue.Queue(), threading.Event()
        self.job_queue = JobQueue()
        self.session_logs = []
        self.active = self.closing = False
        self.operation_cancellable = True
        self.operation_log_job = None
        self.showing_log = False
        root.title('AutoEdit Studio')
        width, height = min(1280, root.winfo_screenwidth() - 80), min(880, root.winfo_screenheight() - 100)
        root.geometry(f'{width}x{height}')
        root.minsize(min(1080, width), min(740, height))
        theme.configure(root)
        self.preset = tk.StringVar(value=PRESETS.get(self.settings['last_preset'], 'Soda Pop'))
        self.output = tk.StringVar(value=self.settings['last_output'])
        self.job_name = tk.StringVar()
        self.status = tk.StringVar(value='Sẵn sàng')
        self.backend_label = tk.StringVar()
        self.queue_hint = tk.StringVar(value='Chưa có job')
        self.result_path = tk.StringVar(value='')
        shell = ttk.Frame(root, padding=(20, 16), style='Body.TFrame')
        shell.pack(fill='both', expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        header = ttk.Frame(shell, style='Body.TFrame')
        header.grid(row=0, column=0, sticky='ew', pady=(0, 16))
        mark = tk.Canvas(header, width=32, height=32, background=theme.BG, highlightthickness=0)
        mark.pack(side='left', padx=(0, 12))
        mark.create_rectangle(1, 1, 31, 31, outline=theme.BORDER)
        mark.create_polygon(6, 24, 12, 8, 17, 8, 11, 24, fill=theme.TEXT)
        mark.create_polygon(16, 24, 22, 8, 27, 8, 21, 24, fill=theme.TEXT)
        ttk.Label(header, text='AUTOEDIT', style='Title.TLabel').pack(side='left')
        self.settings_button = ttk.Button(header, text='Thiết lập', command=self.edit_settings)
        self.settings_button.pack(side='right')
        ttk.Button(header, text='?', width=2, command=self.show_help).pack(side='right', padx=8)
        ttk.Label(header, textvariable=self.backend_label, style='Badge.TLabel').pack(side='right', padx=12)
        metrics = ttk.Frame(shell, style='Panel.TFrame')
        metrics.grid(row=1, column=0, sticky='ew', pady=(0, 14))
        self.counts = {}
        for index, (key, label) in enumerate((('pending', 'Chờ'), ('running', 'Đang chạy'), ('done', 'Hoàn tất'), ('error', 'Lỗi'))):
            metrics.columnconfigure(index, weight=1, uniform='metric')
            card = ttk.Frame(metrics, padding=(18, 8))
            card.grid(row=0, column=index, sticky='ew', padx=1, pady=1)
            self.counts[key] = tk.StringVar(value='0')
            ttk.Label(card, textvariable=self.counts[key], style='Metric.TLabel').pack(side='left', padx=(0, 12))
            ttk.Label(card, text=label, style='Muted.TLabel', font=('Segoe UI', 9)).pack(side='left')
        body = ttk.Frame(shell, style='Body.TFrame')
        body.grid(row=2, column=0, sticky='nsew')
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        builder = ttk.Frame(body, padding=16, width=320, style='Panel.TFrame')
        builder.grid(row=0, column=0, sticky='ns', padx=(0, 12))
        builder.grid_propagate(False)
        builder.columnconfigure(0, weight=1)
        builder.rowconfigure(1, weight=1)
        form_header = ttk.Frame(builder)
        form_header.grid(row=0, column=0, sticky='ew', pady=(0, 14))
        ttk.Label(form_header, text='TẠO JOB', style='Section.TLabel').pack(side='left')
        ttk.Button(form_header, text='Làm mới', command=self.reset_form).pack(side='right')
        viewport = ttk.Frame(builder)
        viewport.grid(row=1, column=0, sticky='nsew')
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)
        canvas = tk.Canvas(viewport, background=theme.PANEL, highlightthickness=0, width=300)
        canvas.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(viewport, orient='vertical', command=canvas.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=scroll.set)
        form = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=form, anchor='nw')
        form.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        self.bind_scroll(self.root, canvas)
        def label(text):
            ttk.Label(form, text=text, style='Muted.TLabel', font=('Segoe UI', 9)).pack(anchor='w', pady=(10, 6))
        label('Tên job')
        ttk.Entry(form, textvariable=self.job_name).pack(fill='x')
        label('Preset')
        ttk.Combobox(form, textvariable=self.preset, values=list(PRESETS.values()), state='readonly').pack(fill='x')
        label('Video nguồn')
        video_box = ttk.Frame(form)
        video_box.pack(fill='x')
        self.videos = tk.Listbox(video_box, height=3, width=20, selectmode='extended', exportselection=False, font=('Segoe UI', 9),
                                 **theme.list_options())
        self.videos.pack(side='left', fill='both', expand=True)
        video_scroll = ttk.Scrollbar(video_box, orient='vertical', command=self.videos.yview)
        video_scroll.pack(side='right', fill='y')
        self.videos.configure(yscrollcommand=video_scroll.set)
        video_x = ttk.Scrollbar(form, orient='horizontal', command=self.videos.xview)
        video_x.pack(fill='x')
        self.videos.configure(xscrollcommand=video_x.set)
        buttons = ttk.Frame(form)
        buttons.pack(fill='x', pady=(8, 0))
        ttk.Button(buttons, text='+ Thêm video', command=self.add_videos).pack(side='left')
        ttk.Button(buttons, text='Bỏ video', command=self.remove_videos).pack(side='right')
        label('Thư mục kết quả')
        folder = ttk.Frame(form)
        folder.pack(fill='x')
        ttk.Entry(folder, textvariable=self.output, width=18).pack(side='left', fill='x', expand=True)
        ttk.Button(folder, text='…', width=3, command=lambda: self.browse_dir(self.output)).pack(side='right', padx=(6, 0))
        label('Ảnh nhân vật')
        ttk.Button(form, text='+ Thêm ảnh', command=self.add_references).pack(anchor='w')
        self.reference_frame = ttk.Frame(form)
        self.reference_frame.pack(fill='x')
        self.refresh_references()
        self.run_button = ttk.Button(builder, text='+ Thêm job', style='Primary.TButton', command=self.run)
        self.run_button.grid(row=2, column=0, sticky='ew', pady=(16, 6))
        workspace = ttk.Frame(body, style='Body.TFrame')
        workspace.grid(row=0, column=1, sticky='nsew')
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        queue_card = ttk.Frame(workspace, padding=16, style='Panel.TFrame')
        queue_card.grid(row=0, column=0, sticky='nsew')
        queue_card.columnconfigure(0, weight=1)
        queue_card.rowconfigure(2, weight=1)
        queue_header = ttk.Frame(queue_card)
        queue_header.grid(row=0, column=0, sticky='ew')
        ttk.Label(queue_header, text='HÀNG ĐỢI', style='Section.TLabel').pack(side='left')
        self.pause_button = ttk.Button(queue_header, text='Tạm dừng hàng đợi', command=self.toggle_pause)
        self.pause_button.pack(side='right')
        self.log_button = ttk.Button(queue_header, text='Nhật ký', command=self.toggle_log)
        self.log_button.pack(side='right', padx=8)
        hint = ttk.Label(queue_card, textvariable=self.queue_hint, style='Muted.TLabel', wraplength=620)
        hint.grid(row=1, column=0, sticky='ew', pady=(8, 12))
        hint.bind('<Configure>', lambda e: hint.configure(wraplength=max(200, e.width)))
        content = ttk.Frame(queue_card)
        content.grid(row=2, column=0, sticky='nsew')
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.table = table = ttk.Frame(content)
        table.grid(row=0, column=0, sticky='nsew')
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.job_list = ttk.Treeview(table, columns=('order', 'name', 'preset', 'state', 'elapsed'), show='headings', selectmode='browse', height=5)
        for key, title, width in (('order', '#', 44), ('name', 'JOB', 180), ('preset', 'PRESET', 90), ('state', 'TRẠNG THÁI', 140), ('elapsed', 'THỜI GIAN', 92)):
            self.job_list.heading(key, text=title, anchor='w')
            self.job_list.column(key, width=width, minwidth=44 if key == 'order' else 70, stretch=key == 'name', anchor='w')
        self.job_list.grid(row=0, column=0, sticky='nsew')
        queue_scroll = ttk.Scrollbar(table, orient='vertical', command=self.job_list.yview)
        queue_scroll.grid(row=0, column=1, sticky='ns')
        self.job_list.configure(yscrollcommand=queue_scroll.set)
        for state, color in (('running', theme.PRIMARY), ('error', theme.DANGER), ('cancelled', theme.MUTED), ('cancelling', theme.WARNING)):
            self.job_list.tag_configure(state, foreground=color)
        self.job_list.tag_configure('running', background='#1d1d1d', font=('Segoe UI Semibold', 10))
        self.job_list.tag_configure('done', foreground=theme.MUTED)
        self.job_list.tag_configure('error', font=('Segoe UI Semibold', 10))
        self.job_list.bind('<<TreeviewSelect>>', self.select_job)
        controls = ttk.Frame(queue_card)
        controls.grid(row=3, column=0, sticky='ew', pady=(12, 0))
        self.remove_button = ttk.Button(controls, text='Hủy job chờ', command=self.cancel_selected, state='disabled')
        self.remove_button.pack(side='left')
        self.cancel_button = ttk.Button(controls, text='Dừng job', style='Danger.TButton', command=self.cancel_active, state='disabled')
        self.cancel_button.pack(side='right')
        self.result_actions = actions = ttk.Frame(queue_card)
        actions.grid(row=4, column=0, sticky='ew', pady=(18, 0))
        ttk.Label(actions, text='Kết quả', style='Muted.TLabel').pack(side='left', padx=(0, 12))
        self.review_button = ttk.Button(actions, text='Xem review', command=self.open_review)
        self.review_button.pack(side='left')
        self.folder_button = ttk.Button(actions, text='Mở thư mục', command=self.open_output)
        self.folder_button.pack(side='left', padx=6)
        self.edits_button = ttk.Button(actions, text='Áp dụng sửa', command=self.apply_edits)
        self.edits_button.pack(side='left')
        self.import_button = ttk.Button(actions, text='Cubase', command=self.import_cubase)
        self.import_button.pack(side='left', padx=6)
        self.log_frame = log_frame = ttk.Frame(content)
        log_frame.grid(row=0, column=0, sticky='nsew')
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=6, width=45, wrap='word', state='disabled', font=('Consolas', 9), **theme.text_options())
        self.log.grid(row=0, column=0, sticky='nsew')
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky='ns')
        self.log.configure(yscrollcommand=log_scroll.set)
        log_frame.grid_remove()
        progress_track = ttk.Frame(shell, height=4, style='Body.TFrame')
        progress_track.grid(row=3, column=0, sticky='ew', pady=(16, 10))
        progress_track.pack_propagate(False)
        self.progress = ttk.Progressbar(progress_track, mode='indeterminate')
        self.progress.pack(fill='both', expand=True)
        ttk.Label(shell, textvariable=self.status, style='Subtitle.TLabel', wraplength=1120).grid(row=4, column=0, sticky='w')
        self.refresh_queue()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Destroy>', self.on_destroy, add='+')
        self.poll_after = self.root.after(100, self.poll)

    def on_destroy(self, event):
        if event.widget is self.root:
            self.root.after_cancel(self.poll_after)

    def show_help(self):
        messagebox.showinfo('Cách dùng',
            'Chọn preset và video, rồi bấm Thêm job. Có thể thêm tiếp khi job khác đang chạy.\n\n'
            'Tên job và ảnh nhân vật có thể để trống. Mỗi job có thư mục kết quả riêng.\n\n'
            'Tạm dừng chỉ giữ các job chờ; Dừng job hủy job đang chạy. '
            'Hàng đợi lưu trong phiên: đóng ứng dụng sẽ hủy các job chưa xong.\n\n'
            'Chọn job để xem nhật ký hoặc mở kết quả. Trong Premiere, tải edit-plan.json qua panel AutoEdit.',
            parent=self.root)

    @staticmethod
    def bind_scroll(window, canvas):
        def scroll(event):
            # Let listboxes/text widgets keep their own scrolling behavior.
            if isinstance(event.widget, (tk.Listbox, tk.Text, ttk.Combobox)):
                return
            if str(event.widget).startswith(str(canvas)):
                canvas.yview_scroll(-int(event.delta / 120), 'units')
        window.bind('<MouseWheel>', scroll, add='+')

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

    def reset_form(self):
        self.job_name.set('')
        self.videos.delete(0, 'end')
        self.references.clear()
        self.refresh_references()

    def toggle_log(self):
        self.showing_log = not self.showing_log
        if self.showing_log:
            self.table.grid_remove()
            self.log_frame.grid()
        else:
            self.log_frame.grid_remove()
            self.table.grid()
        self.log_button.configure(text='Hàng đợi' if self.showing_log else 'Nhật ký')
        self.log_button.configure(style='Active.TButton' if self.showing_log else 'TButton')

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
        for index, path in enumerate(self.references):
            card = ttk.Frame(self.reference_frame)
            card.grid(row=index // 3, column=index % 3, padx=3, pady=6)
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
        dialog.title('AutoEdit · Thiết lập')
        dialog.geometry('960x730')
        dialog.minsize(840, 560)
        dialog.configure(background=theme.BG)
        theme.dark_titlebar(dialog)
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text='Thiết lập áp dụng cho job được thêm tiếp theo. Job đã xếp hàng giữ nguyên cấu hình.',
                  style='Subtitle.TLabel', wraplength=850).pack(anchor='w', padx=24, pady=(18, 0))
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill='both', expand=True, padx=12, pady=12)
        def tab(title):
            pane = ttk.Frame(notebook)
            notebook.add(pane, text=title)
            canvas = tk.Canvas(pane, background=theme.PANEL, highlightthickness=0)
            scroll = ttk.Scrollbar(pane, orient='vertical', command=canvas.yview)
            scroll.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            canvas.configure(yscrollcommand=scroll.set)
            frame = ttk.Frame(canvas, padding=12)
            frame.columnconfigure(1, weight=1)
            window = canvas.create_window((0, 0), window=frame, anchor='nw')
            frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
            canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
            self.bind_scroll(dialog, canvas)
            return frame
        general, advanced = tab('Thiết lập'), tab('Nâng cao')
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
        fields['local_target_seconds'] = entry(advanced, 20, 'Local: mục tiêu tìm nhân vật (giây)', self.settings['local_target_seconds'])
        fields['local_max_seconds'] = entry(advanced, 21, 'Local: giới hạn cứng (0 = tìm đến đủ)', self.settings['local_max_seconds'])
        fields['transition_tail_fallback_sec'] = entry(advanced, 22, 'Các cục trước: chuyển cảnh đuôi (0–2 giây)', self.settings['transition_tail_fallback_sec'])
        fields['source_match_search_budget_sec'] = entry(advanced, 26,
            'Tìm nhanh: ngân sách âm + cảnh (giây, mọi backend)', self.settings['source_match_search_budget_sec'])
        ttk.Label(advanced, text='Ưu tiên âm gốc khớp Voice; đủ cảnh thì dừng. Full quét hết nguồn; xuất file tính riêng.').grid(
            row=27, column=0, columnspan=3, sticky='w')
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
                from autoedit.audio.matching import search_budget
                settings['source_match_search_budget_sec'] = search_budget(settings)
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
                self.refresh_queue()
                dialog.destroy()
            except (ValueError, OSError) as exc:
                messagebox.showerror('Thiết lập', str(exc), parent=dialog)
        ttk.Button(dialog, text='Lưu', command=save).pack(pady=12)

    def setup_local(self, model=None):
        if self.active:
            messagebox.showinfo('Thiết lập local', 'Tạm dừng hàng đợi và chờ job hiện tại kết thúc trước khi cài model.', parent=self.root)
            return
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

    def selected_job(self):
        selection = self.job_list.selection()
        return next((job for job in self.job_queue.jobs if selection and job.id == selection[0]), None)

    def select_job(self, event=None):
        job = self.selected_job()
        self.result_path.set(str(job.output_dir) if job else '')
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        lines = job.logs if job else self.session_logs
        self.log.insert('end', '\n'.join(lines) + ('\n' if lines else ''))
        self.log.see('end')
        self.log.configure(state='disabled')
        self.refresh_actions()

    def refresh_actions(self):
        job = self.selected_job()
        finished = job is not None and job.state == 'done'
        if job is not None and job.state in ('done', 'error', 'cancelled'):
            self.result_actions.grid()
        else:
            self.result_actions.grid_remove()
        for button, enabled in (
                (self.folder_button, job is not None),
                (self.review_button, finished and (job.output_dir / 'ai_review.html').is_file()),
                (self.edits_button, finished and not self.active),
                (self.import_button, finished and not self.active and (job.output_dir / 'edit-plan.json').is_file()),
                (self.remove_button, job is not None and job.state == 'pending'),
                (self.cancel_button, self.active and self.operation_cancellable and not self.cancel.is_set())):
            button.state(['!disabled'] if enabled and not self.closing else ['disabled'])

    def refresh_queue(self):
        states = {'pending': 'Chờ', 'running': '● Đang chạy', 'cancelling': 'Đang dừng',
                  'done': '✓ Xong', 'error': '! Lỗi', 'cancelled': 'Đã hủy'}
        for index, job in enumerate(self.job_queue.jobs, 1):
            values = (f'{index:02d}', job.name, job.preset, states[job.state], job.elapsed())
            if self.job_list.exists(job.id):
                self.job_list.item(job.id, values=values, tags=(job.state,))
            else:
                self.job_list.insert('', 'end', iid=job.id, values=values, tags=(job.state,))
        for key, value in self.counts.items():
            count = sum(job.state == key or (key == 'running' and job.state == 'cancelling') for job in self.job_queue.jobs)
            value.set(str(count))
        pending = sum(job.state == 'pending' for job in self.job_queue.jobs)
        if self.job_queue.paused:
            hint = f'Tạm dừng · {pending} job chờ'
        elif self.job_queue.current:
            hint = f'{pending} job tiếp theo'
        elif self.active:
            hint = f'Đang xử lý · {pending} job chờ'
        elif pending:
            hint = f'{pending} job chờ'
        elif self.job_queue.jobs:
            hint = 'Đã xong'
        else:
            hint = 'Chưa có job'
        self.queue_hint.set(hint)
        self.pause_button.configure(text='Tiếp tục' if self.job_queue.paused else 'Tạm dừng')
        backend = self.settings.get('backend', 'gemini')
        self.backend_label.set(backend.upper())
        self.refresh_actions()

    def append_log(self, text, job=None):
        text = str(text)
        if job is None and self.active and self.job_queue.current is None:
            job = self.operation_log_job
        if job:
            job.logs.append(text)
        else:
            self.session_logs.append(text)
            del self.session_logs[:-1500]
        if self.selected_job() is job:
            self.log.configure(state='normal')
            self.log.insert('end', text + '\n')
            # Bound widget memory as well as the retained history.
            if int(self.log.index('end-1c').split('.')[0]) > 1600:
                self.log.delete('1.0', '101.0')
            self.log.see('end')
            self.log.configure(state='disabled')

    def start(self, operation, job=None, cancellable=True, log_job=None):
        if self.active or self.closing:
            return False
        self.active = True
        self.operation_cancellable = cancellable
        self.operation_log_job = log_job
        self.cancel = job.cancel_event if job else threading.Event()
        self.status.set(f'Đang chạy · {job.name}' if job else 'Đang xử lý…')
        if job is None:
            if log_job is None:
                self.session_logs.clear()
                self.job_list.selection_remove(self.job_list.selection())
                self.select_job()
            if not self.showing_log:
                self.toggle_log()
        self.progress.start(15)
        self.refresh_queue()
        def work():
            result, error = None, None
            try:
                result = operation()
            except Exception as exc:
                error = exc
            self.events.put(('finished', (job.id if job else None, result, error)))
        threading.Thread(target=work, daemon=True).start()
        return True

    def run(self):
        if self.closing:
            return
        key = next(k for k, label in PRESETS.items() if label == self.preset.get())
        try:
            videos = list(self.videos.get(0, 'end'))
            if not videos:
                raise ValueError('Thêm ít nhất một video nguồn trước khi tạo job.')
            for path in videos + list(self.references):
                if not Path(path).is_file():
                    raise ValueError(f'Không tìm thấy file: {path}')
            cfg = apply_preset(self.config, self.settings, key)
            cfg['premiere']['source_media'] = videos
            output = str(Path(self.output.get().strip() or 'output').expanduser().resolve())
            cfg['job']['output_dir'] = output
            cfg.setdefault('video', {}).setdefault('characters', {})['reference_images'] = list(self.references)
            cfg['video']['characters']['reference_regions'] = {}
            settings = {**self.settings, 'last_preset': key, 'last_output': output}
            save_settings(settings)
            name = self.job_name.get().strip() or f'{self.preset.get()} · {Path(videos[0]).stem}'
            job = self.job_queue.add(name[:100], self.preset.get(), cfg)
            self.settings = settings
        except (ValueError, OSError) as exc:
            messagebox.showerror('Không thể tạo job', str(exc), parent=self.root)
            return
        ai = cfg['ai']
        model = ai.get('local_model') if ai.get('backend') == 'local' else ai.get('model')
        job.logs.append(f"{len(videos)} video · {len(self.references)} ảnh · {ai.get('backend')} / {model}")
        job.logs.append('Kết quả: ' + str(job.output_dir))
        self.job_name.set('')
        self.refresh_queue()
        self.job_list.selection_set(job.id)
        self.job_list.see(job.id)
        self.select_job()
        self.dispatch_next()

    def dispatch_next(self):
        if self.active or self.closing:
            return
        job = self.job_queue.start_next()
        if job is None:
            return
        def work():
            from autoedit.worker import run_fresh
            return run_fresh(job.config, log=lambda m: self.events.put(('job_log', (job.id, m))),
                             cancel_event=job.cancel_event)
        self.append_log('Bắt đầu: ' + job.name, job)
        self.start(work, job=job)

    def toggle_pause(self):
        self.job_queue.paused = not self.job_queue.paused
        self.refresh_queue()
        self.dispatch_next()

    def cancel_selected(self):
        job = self.selected_job()
        if job and job.state == 'pending':
            self.job_queue.cancel(job)
            self.refresh_queue()
            self.select_job()

    def cancel_active(self):
        if not self.active or not self.operation_cancellable:
            return
        if self.job_queue.current:
            self.job_queue.cancel(self.job_queue.current)
        else:
            self.cancel.set()
        self.status.set('Đang dừng…')
        self.refresh_queue()
        self.select_job()

    def open_output(self):
        job = self.selected_job()
        if job:
            import os
            os.startfile(str(job.output_dir))

    def open_review(self):
        job = self.selected_job()
        if job:
            path = job.output_dir / 'ai_review.html'
            if path.is_file():
                webbrowser.open(path.resolve().as_uri())

    def apply_edits(self):
        job = self.selected_job()
        if self.active or not job or job.state != 'done':
            return
        path = filedialog.askopenfilename(title='Chọn autoedit-corrections.json', initialdir=str(job.output_dir),
                                          filetypes=[('JSON', '*.json')])
        if not path:
            return
        cfg = {'job': {'output_dir': str(job.output_dir)}, '_ai_review': path}
        def work():
            from autoedit.worker import run_fresh
            run_fresh(cfg, log=lambda m: self.events.put(('log', m)), cancel_event=self.cancel)
        self.start(work, log_job=job)

    def import_cubase(self):
        job = self.selected_job()
        if self.active or not job or job.state != 'done':
            return
        path = job.output_dir / 'edit-plan.json'
        if not path.is_file():
            messagebox.showerror('Cubase', 'Chưa có bộ autofill hoàn chỉnh.')
            return
        def work():
            from autoedit.cubase.host import import_samples
            import_samples(path, log=lambda m: self.events.put(('log', m)))
        # Cubase controls its own host transaction; closing waits for it to finish.
        self.start(work, cancellable=False, log_job=job)

    def poll(self):
        # Drain a bounded batch to keep form input responsive during verbose runs.
        for _ in range(200):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'reload_settings':
                self.settings = load_settings()
                self.refresh_queue()
            elif kind == 'job_log':
                job_id, message = payload
                job = next(job for job in self.job_queue.jobs if job.id == job_id)
                self.append_log(message, job)
            elif kind == 'finished':
                job_id, result, error = payload
                if job_id:
                    job = self.job_queue.current
                    self.job_queue.finish(job, result=result, error=error)
                    message = {'done': 'Hoàn tất.', 'error': 'Lỗi: ' + str(error), 'cancelled': 'Đã hủy.'}[job.state]
                    self.append_log(message, job)
                    summary = {'done': 'Hoàn tất', 'error': 'Lỗi · Xem nhật ký', 'cancelled': 'Đã hủy'}[job.state]
                    self.status.set(f'{summary} · {job.name}')
                else:
                    message = 'Đã hủy.' if self.cancel.is_set() else ('Lỗi: ' + str(error) if error else 'Hoàn tất.')
                    self.append_log(message)
                    self.status.set('Đã hủy' if self.cancel.is_set() else ('Lỗi · Xem nhật ký' if error else 'Hoàn tất'))
                self.active = False
                self.progress.stop()
                self.refresh_queue()
                self.dispatch_next()
            else:
                self.append_log(payload)
        current = self.job_queue.current
        if current:
            self.job_list.set(current.id, 'elapsed', current.elapsed())
        if self.closing and not self.active:
            self.root.destroy()
            return
        self.poll_after = self.root.after(100, self.poll)

    def close(self):
        self.closing = True
        self.job_queue.cancel_all()
        if self.operation_cancellable:
            self.cancel.set()
        self.run_button.state(['disabled'])
        self.pause_button.state(['disabled'])
        self.settings_button.state(['disabled'])
        self.status.set('Đang đóng…')
        self.refresh_queue()


def launch(config_path=None):
    default = ROOT / 'config.yaml'
    cfg = load_config(config_path or (default if default.is_file() else None))
    theme.prepare_dpi()
    root = tk.Tk()
    Application(root, cfg)
    root.mainloop()
    return 0
