import os
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from config import CANCER_TYPES, THRESHOLD
from model_manager import predict_models
from gradcam import generate_gradcam, create_text_image


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CancerDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NeuroScan AI | Diagnostic Support")
        self.geometry("1450x820")
        self.minsize(1250, 760)

        self.results = []
        self.result_cards = []
        self.current_result = None

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_main_view()
        self.build_results_panel()

        self.view_version = 0
        self.stop_event = threading.Event()
        self.is_processing = False

    # -------------------------
    # UI BUILDING
    # -------------------------
    def build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.logo = ctk.CTkLabel(
            self.sidebar,
            text="NeuroScan AI",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo.grid(row=0, column=0, padx=20, pady=30)

        self.type_label = ctk.CTkLabel(self.sidebar, text="Scan Category:")
        self.type_label.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.type_menu = ctk.CTkOptionMenu(self.sidebar, values=CANCER_TYPES)
        self.type_menu.grid(row=2, column=0, padx=20, pady=10)

        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload Scan",
            command=self.open_single_image
        )
        self.upload_btn.grid(row=3, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.batch_upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload Multiple Scans",
            command=self.open_multiple_images
        )
        self.batch_upload_btn.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.folder_upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload Folder",
            command=self.open_folder
        )
        self.folder_upload_btn.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.clear_btn = ctk.CTkButton(
            self.sidebar,
            text="Clear Results",
            fg_color="#444444",
            hover_color="#555555",
            command=self.clear_results
        )
        self.clear_btn.grid(row=6, column=0, padx=20, pady=(30, 10), sticky="ew")

        self.batch_info_label = ctk.CTkLabel(
            self.sidebar,
            text="Batch mode separates scans into suspicious and normal columns.",
            text_color="gray",
            wraplength=170,
            justify="left"
        )
        self.batch_info_label.grid(row=8, column=0, padx=20, pady=20)

        self.stop_btn = ctk.CTkButton(
            self.sidebar,
            text="Stop Batch",
            fg_color="#8b0000",
            hover_color="#a30000",
            command=self.stop_processing,
            state="disabled"
        )
        self.stop_btn.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

    def build_main_view(self):
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_view.grid_rowconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(self.main_view, width=650, height=540)
        self.tabs.grid(row=0, column=0, pady=15, sticky="nsew")

        self.tab_original = self.tabs.add("Original Scan")
        self.tab_gradcam = self.tabs.add("Grad-CAM")

        self.original_label = ctk.CTkLabel(self.tab_original, text="No scan uploaded")
        self.original_label.pack(expand=True, padx=20, pady=20)

        self.gradcam_label = ctk.CTkLabel(self.tab_gradcam, text="No analysis yet")
        self.gradcam_label.pack(expand=True, padx=20, pady=20)

        self.result_card = ctk.CTkFrame(self.main_view, fg_color="#2b2b2b")
        self.result_card.grid(row=1, column=0, padx=40, pady=20)

        self.status_label = ctk.CTkLabel(
            self.result_card,
            text="SYSTEM READY",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.status_label.pack(pady=(10, 5))

        self.detail_label = ctk.CTkLabel(
            self.result_card,
            text="Awaiting input...",
            text_color="gray",
            wraplength=800
        )
        self.detail_label.pack(pady=(0, 10))

        self.progress = ctk.CTkProgressBar(self.main_view)
        self.progress.set(0)
        # Important: main_view uses grid(), so the progress bar must also use grid().
        # We hide/show it with grid_remove()/grid().
        self.progress.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")
        self.progress.grid_remove()

    def build_results_panel(self):
        self.results_panel = ctk.CTkFrame(self, width=410, fg_color="#1f1f1f")
        self.results_panel.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        self.results_panel.grid_propagate(False)

        title = ctk.CTkLabel(
            self.results_panel,
            text="Batch Classification Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(padx=15, pady=(15, 5))

        self.summary_label = ctk.CTkLabel(
            self.results_panel,
            text="Cancer: 0 | Normal: 0",
            text_color="gray"
        )
        self.summary_label.pack(padx=15, pady=(0, 10))

        columns_frame = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        columns_frame.pack(fill="both", expand=True, padx=10, pady=10)
        columns_frame.grid_columnconfigure(0, weight=1)
        columns_frame.grid_columnconfigure(1, weight=1)
        columns_frame.grid_rowconfigure(1, weight=1)

        cancer_title = ctk.CTkLabel(
            columns_frame,
            text="Cancer / Suspicious",
            text_color="#ff6b6b",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        cancer_title.grid(row=0, column=0, padx=5, pady=(0, 8), sticky="ew")

        normal_title = ctk.CTkLabel(
            columns_frame,
            text="No Cancer / Normal",
            text_color="#66ff66",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        normal_title.grid(row=0, column=1, padx=5, pady=(0, 8), sticky="ew")

        self.cancer_scroll = ctk.CTkScrollableFrame(columns_frame, width=185, fg_color="#2a2020")
        self.cancer_scroll.grid(row=1, column=0, padx=5, sticky="nsew")

        self.normal_scroll = ctk.CTkScrollableFrame(columns_frame, width=185, fg_color="#202a20")
        self.normal_scroll.grid(row=1, column=1, padx=5, sticky="nsew")

    def reset_display_label(self, label_name, parent, text):
        old_label = getattr(self, label_name)

        try:
            old_label.destroy()
        except Exception:
            pass

        new_label = ctk.CTkLabel(parent, text=text)
        new_label.pack(expand=True, padx=20, pady=20)

        setattr(self, label_name, new_label)

    def stop_processing(self):
        self.stop_event.set()
        self.status_label.configure(
            text="STOPPING...",
            text_color="#ffcc00"
        )
        self.detail_label.configure(
            text="Stopping after the current scan finishes. Already processed results will stay."
        )
        self.stop_btn.configure(state="disabled")

    # -------------------------
    # FILE SELECTION
    # -------------------------
    def open_single_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.process_files([file_path], clear_existing=False)

    def open_multiple_images(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_paths:
            self.process_files(list(file_paths), clear_existing=True)

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        file_paths = []

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in valid_ext:
                    file_paths.append(os.path.join(root, file_name))

        if not file_paths:
            messagebox.showinfo("No Images Found", "No supported image files were found in this folder.")
            return

        self.process_files(file_paths, clear_existing=True)

    # -------------------------
    # PROCESSING
    # -------------------------
    def process_files(self, file_paths, clear_existing=True):
        self.stop_event.clear()
        self.is_processing = True

        if clear_existing:
            self.clear_results()

        cancer_type = self.type_menu.get()

        self.set_controls_state("disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="ANALYZING BATCH...", text_color="#3b8ed0")
        self.detail_label.configure(text=f"Processing {len(file_paths)} scan(s)...")
        self.reset_display_label(
            "gradcam_label",
            self.tab_gradcam,
            "Select a result to generate Grad-CAM"
        )

        self.progress.grid()
        self.progress.start()

        thread = threading.Thread(
            target=self.worker_batch_analyze,
            args=(file_paths, cancer_type),
            daemon=True
        )
        thread.start()

    def worker_batch_analyze(self, file_paths, cancer_type):
        total = len(file_paths)

        for index, file_path in enumerate(file_paths, start=1):

            if self.stop_event.is_set():
                break

            try:
                score, predictions, best_classical_item = predict_models(file_path, cancer_type)

                if score is None:
                    result = {
                        "file_path": file_path,
                        "cancer_type": cancer_type,
                        "score": None,
                        "predictions": [],
                        "best_model": "No model available",
                        "best_classical_item": None,
                        "label": "unknown",
                        "gradcam_image": None,
                    }
                else:
                    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
                    best = sorted_preds[0]
                    label = "cancer" if score >= THRESHOLD else "normal"

                    result = {
                        "file_path": file_path,
                        "cancer_type": cancer_type,
                        "score": score,
                        "predictions": predictions,
                        "best_model": best["model"],
                        "best_classical_item": best_classical_item,
                        "label": label,
                        "gradcam_image": None,
                    }

                self.after(0, self.add_result_card, result)
                self.after(
                    0,
                    self.detail_label.configure,
                    {"text": f"Processed {index}/{total}: {os.path.basename(file_path)}"}
                )

            except Exception as e:
                error_result = {
                    "file_path": file_path,
                    "cancer_type": cancer_type,
                    "score": None,
                    "predictions": [],
                    "best_model": f"Error: {e}",
                    "best_classical_item": None,
                    "label": "unknown",
                    "gradcam_image": create_text_image("Analysis error"),
                }
                self.after(0, self.add_result_card, error_result)

        self.after(0, self.finish_batch)

    def finish_batch(self):
        self.progress.stop()
        self.progress.grid_remove()

        self.render_results_sorted()

        self.is_processing = False
        was_stopped = self.stop_event.is_set()

        self.set_controls_state("normal")
        self.stop_btn.configure(state="disabled")

        cancer_count = sum(1 for r in self.results if r["label"] == "cancer")
        normal_count = sum(1 for r in self.results if r["label"] == "normal")
        unknown_count = sum(1 for r in self.results if r["label"] == "unknown")

        if was_stopped:
            self.status_label.configure(
                text="BATCH STOPPED",
                text_color="#ffcc00"
            )
            self.detail_label.configure(
                text=f"Stopped by user. Kept processed results: Cancer/Suspicious: {cancer_count} | No Cancer/Normal: {normal_count} | Unknown/Error: {unknown_count}"
            )
        else:
            self.status_label.configure(
                text="BATCH ANALYSIS COMPLETE",
                text_color="#3b8ed0"
            )
            self.detail_label.configure(
                text=f"Cancer/Suspicious: {cancer_count} | No Cancer/Normal: {normal_count} | Unknown/Error: {unknown_count}"
            )

    # -------------------------
    # RESULT PANEL
    # -------------------------
    def add_result_card(self, result):
        self.results.append(result)
        self.update_summary()

        if self.current_result is None:
            self.select_result(result)

    def render_results_sorted(self):
        for widget in self.cancer_scroll.winfo_children():
            widget.destroy()
        for widget in self.normal_scroll.winfo_children():
            widget.destroy()

        cancer_results = [r for r in self.results if r["label"] == "cancer"]
        normal_results = [r for r in self.results if r["label"] == "normal"]
        unknown_results = [r for r in self.results if r["label"] == "unknown"]

        cancer_results.sort(key=lambda r: r["score"] or 0, reverse=True)

        # For normal cases, lowest tumor probability = safest normal
        normal_results.sort(key=lambda r: r["score"] if r["score"] is not None else 1)

        for result in cancer_results:
            self.create_result_card(result, self.cancer_scroll)

        for result in normal_results:
            self.create_result_card(result, self.normal_scroll)

        for result in unknown_results:
            self.create_result_card(result, self.cancer_scroll)

    def create_result_card(self, result, parent):
        card_color = "#3a2424" if result["label"] == "cancer" else "#243a24"
        if result["label"] == "unknown":
            card_color = "#333333"

        card = ctk.CTkFrame(parent, fg_color=card_color, corner_radius=10)
        card.pack(fill="x", padx=5, pady=6)

        thumb_img = self.load_thumbnail(result["file_path"], size=(110, 80))

        img_btn = ctk.CTkButton(
            card,
            text="",
            image=thumb_img,
            width=120,
            height=85,
            fg_color="transparent",
            hover_color="#444444",
            command=lambda r=result: self.select_result(r)
        )
        img_btn.image = thumb_img
        img_btn.pack(padx=6, pady=(6, 3))

        file_name = os.path.basename(result["file_path"])
        if len(file_name) > 22:
            file_name = file_name[:19] + "..."

        name_label = ctk.CTkLabel(
            card,
            text=file_name,
            font=ctk.CTkFont(size=11),
            wraplength=150
        )
        name_label.pack(padx=5, pady=(0, 2))

        score_text = "Score: N/A" if result["score"] is None else f"Score: {result['score']:.1%}"

        score_label = ctk.CTkLabel(
            card,
            text=score_text,
            text_color="#ffb3b3" if result["label"] == "cancer" else "#b3ffb3",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        score_label.pack(padx=5, pady=(0, 6))

    def update_summary(self):
        cancer_count = sum(1 for r in self.results if r["label"] == "cancer")
        normal_count = sum(1 for r in self.results if r["label"] == "normal")
        unknown_count = sum(1 for r in self.results if r["label"] == "unknown")

        text = f"Cancer: {cancer_count} | Normal: {normal_count}"
        if unknown_count:
            text += f" | Unknown/Error: {unknown_count}"

        self.summary_label.configure(text=text)

    def select_result(self, result):
        self.current_result = result
        self.show_original_image(result["file_path"])

        if result["score"] is None:
            self.status_label.configure(text="NO RESULT AVAILABLE", text_color="#ffcc00")
            self.detail_label.configure(text=result["best_model"])
            self.reset_display_label(
                "gradcam_label",
                self.tab_gradcam,
                "Grad-CAM unavailable"
            )
            return

        if result["label"] == "cancer":
            result["gradcam_image"] = None
            self.status_label.configure(text=f"PRIORITY ALERT: {result['score']:.1%}", text_color="#ff4b4b")
            self.detail_label.configure(text=f"Highest abnormality score from: {result['best_model']}")
        else:
            self.status_label.configure(text=f"SCAN ANALYSIS: NORMAL ({result['score']:.1%})", text_color="#46ff46")
            self.detail_label.configure(text=f"No tumor found. Highest score from: {result['best_model']}")

        self.show_or_generate_gradcam(result)

    # -------------------------
    # IMAGE / GRADCAM HELPERS
    # -------------------------
    def pil_to_ctk(self, pil_img, max_size=(500, 500)):
        img = pil_img.copy().convert("RGB")
        img.thumbnail(max_size)

        return ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=img.size
        )

    def load_thumbnail(self, file_path, size=(110, 80)):
        img = Image.open(file_path).convert("RGB")
        img.thumbnail(size)
        return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

    def show_original_image(self, file_path):
        img = Image.open(file_path).convert("RGB")
        tk_img = self.pil_to_ctk(img)

        self.original_label.configure(image=tk_img, text="")
        self.original_label.image = tk_img

    def show_or_generate_gradcam(self, result):
        if result.get("gradcam_image") is not None:
            self.show_gradcam_image(result["gradcam_image"])
            return

        self.reset_display_label(
            "gradcam_label",
            self.tab_gradcam,
            "Preparing Grad-CAM..."
        )

        thread = threading.Thread(
            target=self.worker_generate_gradcam,
            args=(result,),
            daemon=True
        )
        thread.start()

    def worker_generate_gradcam(self, result):
        try:
            if result["score"] is None:
                gradcam_image = create_text_image("Grad-CAM unavailable")

            elif result["score"] >= THRESHOLD:
                if result["best_classical_item"] is not None:
                    gradcam_image = generate_gradcam(
                        result["file_path"],
                        result["best_classical_item"]
                    )

                    if gradcam_image is None:
                        gradcam_image = create_text_image("Grad-CAM unavailable")
                else:
                    gradcam_image = create_text_image("No classical model available for Grad-CAM")

            else:
                gradcam_image = create_text_image("No tumor found")

            result["gradcam_image"] = gradcam_image
            self.after(0, self.show_gradcam_image, gradcam_image)

        except Exception as e:
            error_img = create_text_image(f"Grad-CAM error: {e}")
            result["gradcam_image"] = error_img
            self.after(0, self.show_gradcam_image, error_img)

    def show_gradcam_image(self, gradcam_image):
        tk_gradcam = self.pil_to_ctk(gradcam_image)
        self.gradcam_label.configure(image=tk_gradcam, text="")
        self.gradcam_label.image = tk_gradcam

    # -------------------------
    # UTILS
    # -------------------------
    def clear_results(self):
        self.results.clear()
        self.result_cards.clear()
        self.current_result = None

        for widget in self.cancer_scroll.winfo_children():
            widget.destroy()
        for widget in self.normal_scroll.winfo_children():
            widget.destroy()

        self.summary_label.configure(text="Cancer: 0 | Normal: 0")
        self.status_label.configure(text="SYSTEM READY", text_color="white")
        self.detail_label.configure(text="Awaiting input...")

        self.view_version += 1

        self.reset_display_label(
            "original_label",
            self.tab_original,
            "No scan uploaded"
        )

        self.reset_display_label(
            "gradcam_label",
            self.tab_gradcam,
            "No analysis yet"
        )

    def set_controls_state(self, state):
        self.upload_btn.configure(state=state)
        self.batch_upload_btn.configure(state=state)
        self.folder_upload_btn.configure(state=state)
        self.clear_btn.configure(state=state)

    def show_error(self, error_message):
        self.progress.stop()
        self.progress.grid_remove()
        self.set_controls_state("normal")

        self.status_label.configure(text="ERROR", text_color="#ff4b4b")
        self.detail_label.configure(text=error_message)
        messagebox.showerror("Analysis Error", error_message)


if __name__ == "__main__":
    app = CancerDetectionApp()
    app.mainloop()
