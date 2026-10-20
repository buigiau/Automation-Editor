"""Review detected reference regions and select individual character crops."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox


def select_regions(parent, path, source_kind, existing=None):
    from PIL import Image, ImageTk
    import cv2
    import numpy as np
    from autoedit.video.faces import MultiRegionFaceLandmarker
    picture = Image.open(path).convert('RGB')
    width, height = picture.size
    scale = min(1., 840/width, 520/height)
    dialog = tk.Toplevel(parent)
    dialog.title('Select target characters: ' + Path(path).name)
    dialog.transient(parent)
    chosen = [dict(region) for region in (existing or [])]
    kind = tk.StringVar(value='body' if source_kind == 'animation' else 'face')
    ttk.Label(dialog, text='Drag a box around ONE character. Repeat for more targets. Click a box to remove it.').pack(padx=12, pady=8)
    controls = ttk.Frame(dialog)
    controls.pack(fill=tk.X, padx=12)
    ttk.Radiobutton(controls, text='Whole character / animated vehicle', variable=kind, value='body').pack(side=tk.LEFT)
    ttk.Radiobutton(controls, text='Readable face', variable=kind, value='face').pack(side=tk.LEFT, padx=12)
    canvas = tk.Canvas(dialog, width=round(width*scale), height=round(height*scale))
    canvas.pack(padx=12, pady=8)
    preview = ImageTk.PhotoImage(picture.resize((round(width*scale),round(height*scale))))
    canvas.create_image(0,0,image=preview,anchor='nw')
    status = tk.StringVar()
    ttk.Label(dialog, textvariable=status).pack()
    rectangles = []
    def redraw():
        for item in rectangles:
            canvas.delete(item)
        rectangles.clear()
        for region in chosen:
            x,y,w,h=region['box']
            rectangles.append(canvas.create_rectangle(x*scale,y*scale,(x+w)*scale,(y+h)*scale,
                                                      outline='#32cf82',width=3))
        status.set(f'{len(chosen)} targets selected')
    drag = {}
    def down(event):
        drag.update(x=max(0,min(event.x,width*scale)),y=max(0,min(event.y,height*scale)))
        drag['item'] = canvas.create_rectangle(drag['x'],drag['y'],drag['x'],drag['y'],outline='orange',width=2)
    def move(event):
        if 'item' in drag:
            canvas.coords(drag['item'],drag['x'],drag['y'],max(0,min(event.x,width*scale)),max(0,min(event.y,height*scale)))
    def up(event):
        if 'item' not in drag:
            return
        canvas.delete(drag['item'])
        x0,y0,x1,y1=drag['x']/scale,drag['y']/scale,max(0,min(event.x/scale,width)),max(0,min(event.y/scale,height))
        drag.clear()
        if abs(x1-x0) < 10 and abs(y1-y0) < 10:
            for index,region in enumerate(chosen):
                x,y,w,h=region['box']
                if x <= x1 <= x+w and y <= y1 <= y+h:
                    chosen.pop(index)
                    break
        elif min(abs(x1-x0),abs(y1-y0)) >= 32:
            chosen.append({'box':[round(min(x0,x1)),round(min(y0,y1)),round(abs(x1-x0)),round(abs(y1-y0))], 'kind':kind.get()})
        redraw()
    canvas.bind('<ButtonPress-1>',down)
    canvas.bind('<B1-Motion>',move)
    canvas.bind('<ButtonRelease-1>',up)
    result = []
    def accept():
        if not chosen:
            messagebox.showerror('Target characters','Select at least one character region.',parent=dialog)
            return
        result.extend(chosen)
        dialog.destroy()
    buttons = ttk.Frame(dialog)
    buttons.pack(pady=10)
    ttk.Button(buttons,text='Clear boxes',command=lambda:(chosen.clear(),redraw())).pack(side=tk.LEFT,padx=6)
    ttk.Button(buttons,text='Use selected targets',command=accept).pack(side=tk.LEFT,padx=6)
    ttk.Button(buttons,text='Cancel',command=dialog.destroy).pack(side=tk.LEFT,padx=6)
    if not chosen:
        detector = MultiRegionFaceLandmarker(include_animation=source_kind == 'animation')
        try:
            frame = cv2.cvtColor(np.asarray(picture),cv2.COLOR_RGB2BGR)
            # Suggestions are bounded and do not occupy the GPU job queue.
            small = cv2.resize(frame,(round(width*scale),round(height*scale)))
            faces = detector.detect(small,1).get('faces', [])
            for face in faces:
                x,y,w,h = [round(v/scale) for v in face['face']]
                x,y = max(0,x),max(0,y)
                chosen.append({'box':[x,y,min(w,width-x),min(h,height-y)], 'kind':'face'})
        finally:
            detector.close()
    redraw()
    dialog.grab_set()
    parent.wait_window(dialog)
    return result or None
