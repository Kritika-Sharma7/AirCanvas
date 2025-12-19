import cv2
import numpy as np
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image
import time
import os


def save_canvas_as_pdf(canvas_img):
    """
    canvas_img: NumPy array (OpenCV image)
    """

    # -------------------------------
    # Convert OpenCV image → PIL Image
    # -------------------------------
    rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # -------------------------------
    # Create filename
    # -------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"AirCanvas_{timestamp}.pdf"

    # -------------------------------
    # Create PDF
    # -------------------------------
    pdf = pdf_canvas.Canvas(filename, pagesize=A4)
    page_width, page_height = A4

    img_width, img_height = pil_img.size

    # Maintain aspect ratio
    scale = min(page_width / img_width, page_height / img_height)
    draw_width = img_width * scale
    draw_height = img_height * scale

    x = (page_width - draw_width) / 2
    y = (page_height - draw_height) / 2

    pdf.drawImage(
        ImageReader(pil_img),
        x,
        y,
        width=draw_width,
        height=draw_height
    )

    pdf.showPage()
    pdf.save()

    print(f"[✔] PDF saved as {filename}")
