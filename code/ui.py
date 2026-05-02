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
        self.geometry("1100x760")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

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
            command=self.open_and_process
        )
        self.upload_btn.grid(row=3, column=0, padx=20, pady=20)

        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.tabs = ctk.CTkTabview(self.main_view, width=650, height=520)
        self.tabs.pack(pady=15, expand=True)

        self.tab_original = self.tabs.add("Original Scan")
        self.tab_gradcam = self.tabs.add("Grad-CAM")

        self.original_label = ctk.CTkLabel(self.tab_original, text="No scan uploaded")
        self.original_label.pack(expand=True, padx=20, pady=20)

        self.gradcam_label = ctk.CTkLabel(self.tab_gradcam, text="No analysis yet")
        self.gradcam_label.pack(expand=True, padx=20, pady=20)

        self.result_card = ctk.CTkFrame(self.main_view, fg_color="#2b2b2b")
        self.result_card.pack(fill="x", padx=40, pady=20)

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

    def open_and_process(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        cancer_type = self.type_menu.get()

        self.show_original_image(file_path)
        self.gradcam_label.configure(image=None, text="Analysis running...")

        self.status_label.configure(
            text="ANALYZING...",
            text_color="#3b8ed0"
        )

        self.detail_label.configure(
            text="Running classical and hybrid ensemble models..."
        )

        self.progress.pack(pady=10)
        self.progress.start()
        self.upload_btn.configure(state="disabled")

        thread = threading.Thread(
            target=self.worker_analyze,
            args=(file_path, cancer_type),
            daemon=True
        )
        thread.start()

    def pil_to_ctk(self, pil_img, max_size=(500, 500)):
        img = pil_img.copy().convert("RGB")
        img.thumbnail(max_size)

        return ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=img.size
        )

    def show_original_image(self, file_path):
        img = Image.open(file_path).convert("RGB")
        tk_img = self.pil_to_ctk(img)

        self.original_label.configure(image=tk_img, text="")
        self.original_label.image = tk_img

    def worker_analyze(self, file_path, cancer_type):
        try:
            score, predictions, best_classical_item = predict_models(file_path, cancer_type)

            gradcam_image = None

            if score is not None and score >= THRESHOLD:
                gradcam_image = generate_gradcam(file_path, best_classical_item)

                if gradcam_image is None:
                    gradcam_image = create_text_image("Grad-CAM unavailable")
            else:
                gradcam_image = create_text_image("No tumor found")

            self.after(
                0,
                self.update_result,
                score,
                predictions,
                gradcam_image
            )

        except Exception as e:
            self.after(0, self.show_error, str(e))

    def update_result(self, score, predictions, gradcam_image):
        self.progress.stop()
        self.progress.pack_forget()
        self.upload_btn.configure(state="normal")

        if score is None:
            self.status_label.configure(
                text="NO MODEL AVAILABLE",
                text_color="#ffcc00"
            )
            self.detail_label.configure(
                text="No model was loaded for this cancer type."
            )
            return

        sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
        best = sorted_preds[0]

        if score >= THRESHOLD:
            self.status_label.configure(
                text=f"PRIORITY ALERT: {score:.1%}",
                text_color="#ff4b4b"
            )
            self.detail_label.configure(
                text=f"Highest abnormality score from: {best['model']}"
            )
        else:
            self.status_label.configure(
                text=f"SCAN ANALYSIS: NORMAL ({score:.1%})",
                text_color="#46ff46"
            )
            self.detail_label.configure(
                text=f"No tumor found. Highest score from: {best['model']}"
            )

        tk_gradcam = self.pil_to_ctk(gradcam_image)
        self.gradcam_label.configure(image=tk_gradcam, text="")
        self.gradcam_label.image = tk_gradcam

    def show_error(self, error_message):
        self.progress.stop()
        self.progress.pack_forget()
        self.upload_btn.configure(state="normal")

        self.status_label.configure(
            text="ERROR",
            text_color="#ff4b4b"
        )

        self.detail_label.configure(text=error_message)
        messagebox.showerror("Analysis Error", error_message)