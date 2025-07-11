import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk

BG_COLOR = "#f4f4f4"
BTN_COLOR = "#4CAF50"
BTN_TEXT_COLOR = "#ffffff"
ENTRY_BG = "#ffffff"
FONT_NAME = "Segoe UI"

def launch_gui(predict_weather_fn):
    def submit():
        try:
            tavg = float(entry_tavg.get())
            tmin = float(entry_tmin.get())
            tmax = float(entry_tmax.get())
            prcp = float(entry_prcp.get())

            # Validation Rules
            if not (-50 <= tmin <= tmax <= 60):
                raise ValueError("Temperatures must be between -50°C and 60°C with tmin ≤ tmax.")
            if not (0 <= prcp <= 1000):
                raise ValueError("Precipitation must be between 0 and 1000 mm.")

            result = predict_weather_fn(tavg, tmin, tmax, prcp)
            messagebox.showinfo("Prediction", f"Predicted weather: {result.upper()}")

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    # Load and resize background image

    width, height = 400, 660

    root = tk.Tk()
    root.title("🌤️ AskWeather - Weather Classifier")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)

    # Resizing the image
    bg_image_raw = Image.open("assets/weather_banner.jpg")
    bg_image_resized = bg_image_raw.resize((400, 660), Image.Resampling.LANCZOS)
    bg_image = ImageTk.PhotoImage(bg_image_resized)

    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")

    canvas.create_rectangle(30, 20, 370, 120, fill="#ffffff", stipple="gray50", outline="")

    title_font = tkFont.Font(family=FONT_NAME, size=16, weight="bold")
    label_font = tkFont.Font(family=FONT_NAME, size=11)
    entry_font = tkFont.Font(family=FONT_NAME, size=10)

    title = tk.Label(root, text="AskWeather 🌤️", font=title_font, bg="#ffffff")
    canvas.create_window(width // 2, 50, window=title)

    form_frame = tk.Frame(root, bg="#ffffff", bd=0)
    canvas.create_window(width // 2, height // 2, window=form_frame)

    labels = ["Avg Temp (tavg)", "Min Temp (tmin)", "Max Temp (tmax)", "Precipitation (prcp)"]
    entries = []

    for i, text in enumerate(labels):
        lbl = tk.Label(form_frame, text=text, font=label_font, bg="#ffffff")
        lbl.grid(row=i, column=0, padx=10, pady=5, sticky="e")

        entry = tk.Entry(form_frame, font=entry_font, width=20)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(entry)

    entry_tavg, entry_tmin, entry_tmax, entry_prcp = entries

    predict_btn = tk.Button(root, text="Predict Weather 🌱", font=label_font,
                            bg="#4CAF50", fg="white", activebackground="#45a049",
                            command=submit, padx=20, pady=8)

    canvas.create_window(width // 2, height - 60, window=predict_btn)

    root.mainloop()