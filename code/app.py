from model_manager import load_all_models
from ui import CancerDetectionApp


if __name__ == "__main__":
    load_all_models()
    app = CancerDetectionApp()
    app.mainloop()