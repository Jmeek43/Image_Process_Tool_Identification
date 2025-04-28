import cv2
import pickle
from  KNNClassifier import KNNClassifier

import numpy as np
import tkinter as tk

from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageDraw
from scipy.ndimage import zoom as ndimage_zoom, sobel, gaussian_filter
from tkinter import filedialog, messagebox

# Predefined convolution kernels for different PSF effects
KERNELS = {
    "Gaussian Blur": np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]) / 256,

    "Unsharp Mask (Sharpen)": np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]),

    "Edge Detection (Laplacian)": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]),

    "Embossing": np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]),

    "Motion Blur": np.diag([1/9] * 9)
}


class PSFViewer:
    def __init__(self, root):
        # Initialize the main window and UI components
        self.root = root
        self.path = ""
        self.root.title("Image Viewer with PSF Effects")

        # Setup frames and canvases for displaying images
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.left_canvas = tk.Label(self.canvas_frame)
        self.left_canvas.pack(side=tk.LEFT, padx=10)

        self.right_canvas = tk.Label(self.canvas_frame)
        self.right_canvas.pack(side=tk.LEFT, padx=10)
        self.right_canvas.bind("<Button-1>", self.magnify_at_click)

        # Setup controls
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        # Load classifier and scaler
        self.classifier = KNNClassifier()
        self.model, self.scaler = self.load_model('svm_pca_model.pkl')
        self.classifier.model = self.model
        self.classifier.scaler = self.scaler

        # Buttons and effect dropdown
        self.browse_btn = tk.Button(self.control_frame, text="Browse Image", command=self.browse_image)
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        all_effects = list(KERNELS.keys()) + [
            "Greyscale",
            "Edge Detection (Sobel)",
            "Canny Edge Detection",
            "Texture Spectrum",
            "Critical Point Detection",
            "Shape Detection"
        ]
        self.effect_var = tk.StringVar()
        self.effect_menu = tk.OptionMenu(self.control_frame, self.effect_var, *all_effects, command=self.on_effect_change)
        self.effect_var.set("Gaussian Blur")
        self.effect_menu.pack(side=tk.LEFT, padx=5)

        self.apply_btn = tk.Button(self.control_frame, text="Apply PSF", command=self.apply_psf)
        self.apply_btn.pack(side=tk.LEFT, padx=5)

        self.sharpness_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Canny Threshold", command=self.on_slider_change)
        self.sharpness_slider.set(100)
        self.sharpness_slider.pack(side=tk.LEFT, padx=5)
        self.sharpness_slider.pack_forget()  # Hidden by default

        self.classify_btn = tk.Button(self.control_frame, text="Classify Image", command=self.classify_image)
        self.classify_btn.pack(side=tk.LEFT, padx=5)

        self.capture_btn = tk.Button(self.control_frame, text="Capture Photo", command=self.capture_image)
        if self.check_camera():
            self.capture_btn.config(state=tk.NORMAL)
        else:
            self.capture_btn.config(state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.original_image = None
        self.processed_image = None

    def on_effect_change(self, selection):
        # Show or hide sharpness slider depending on effect selected
        if selection == "Canny Edge Detection" or selection == "Critical Point Detection":
            self.sharpness_slider.pack(side=tk.LEFT, padx=5)
        else:
            self.sharpness_slider.pack_forget()

    def on_slider_change(self, value):
        # Reapply PSF when the slider is changed for Canny/critical points
        if self.effect_var.get() in ["Canny Edge Detection", "Critical Point Detection"]:
            self.apply_psf()

    def browse_image(self):
        # Open a file dialog to browse and load an image
        self.path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif")])
        if self.path:
            self.original_image = Image.open(self.path).convert("RGB")
            self.processed_image = self.original_image.copy()
            self.display_images()

    def apply_psf(self):
        # Apply selected PSF or image processing effect to the image
        if self.original_image is None:
            return

        selected = self.effect_var.get()
        img = self.original_image

        if selected == "Greyscale":
            self.processed_image = ImageOps.grayscale(img).convert("RGB")

        elif selected == "Edge Detection (Sobel)":
            # Apply Sobel Edge detection
            gray = np.array(ImageOps.grayscale(img), dtype=float)
            dx = sobel(gray, axis=0)
            dy = sobel(gray, axis=1)
            edges = np.hypot(dx, dy)
            edges = (255 * edges / np.max(edges)).astype(np.uint8)
            self.processed_image = Image.fromarray(edges).convert("RGB")

        elif selected == "Canny Edge Detection":
            gray = np.array(ImageOps.grayscale(img))
            threshold1 = self.sharpness_slider.get()
            threshold2 = threshold1 * 2
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.processed_image = Image.fromarray(edges).convert("RGB")

        elif selected == "Texture Spectrum":
            gray = np.array(ImageOps.grayscale(img), dtype=np.uint8)
            tex_img = self.texture_spectrum(gray)
            self.processed_image = Image.fromarray(tex_img).convert("RGB")

        elif selected == "Critical Point Detection":
            gray = np.array(ImageOps.grayscale(img))
            threshold1 = self.sharpness_slider.get()
            threshold2 = threshold1 * 2
            edges = cv2.Canny(gray, threshold1, threshold2)
            corners = cv2.goodFeaturesToTrack(edges, maxCorners=200, qualityLevel=0.01, minDistance=10)
            img_with_circles = img.copy()
            draw = ImageDraw.Draw(img_with_circles)
            if corners is not None:
                for pt in corners:
                    x, y = pt.ravel()
                    bbox = [x - 5, y - 5, x + 5, y + 5]
                    draw.ellipse(bbox, outline="red", width=2)
            self.processed_image = img_with_circles

        elif selected == "Shape Detection":
            self.processed_image = self.detect_shapes()

        elif selected in KERNELS:
            # Apply convolution kernel effect
            kernel = KERNELS[selected]
            kernel_size = kernel.shape[0]
            pil_kernel = ImageFilter.Kernel(
                size=(kernel_size, kernel_size),
                kernel=kernel.flatten(),
                scale=np.sum(kernel) if np.sum(kernel) != 0 else 1
            )
            self.processed_image = img.filter(pil_kernel)

        self.display_images()

    def texture_spectrum(self, gray_img):
        # Calculate texture spectrum based on pixel neighborhoods
        h, w = gray_img.shape
        output = np.zeros((h - 2, w - 2), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = gray_img[y, x]
                patch = gray_img[y - 1:y + 2, x - 1:x + 2].flatten()
                bits = (patch >= center).astype(np.uint8)
                bits = np.delete(bits, 4)
                value = sum([bit << i for i, bit in enumerate(bits)])
                output[y - 1, x - 1] = value

        return output

    def critical_points(self, gray, num_points=200, radius=5):
        # Detect critical points and display in the image
        gray_blur = gaussian_filter(gray, sigma=1)

        Ixx = gaussian_filter(gray_blur, sigma=1, order=(2, 0))
        Iyy = gaussian_filter(gray_blur, sigma=1, order=(0, 2))
        Ixy = gaussian_filter(gray_blur, sigma=1, order=(1, 1))

        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        R = det - 0.04 * (trace ** 2)

        R_norm = (R - R.min()) / (R.max() - R.min())
        flat_indices = np.argsort(R_norm.ravel())[::-1]
        h, w = R.shape
        keypoints = [(i % w, i // w) for i in flat_indices[:num_points]]

        img_with_circles = self.original_image.copy()
        draw = ImageDraw.Draw(img_with_circles)
        for x, y in keypoints:
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline="red", width=2)

        return img_with_circles

    def detect_shapes(self):
        # Detect and label basic shapes in the image
        img = np.array(self.original_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        threshold_type = cv2.THRESH_BINARY_INV if mean_val > 127 else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        labeled = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)

            label = "Unknown"
            vertices = len(approx)
            aspect_ratio = w / float(h)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)

            if vertices == 3:
                label = "Triangle"
            elif vertices == 4:
                label = "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
            elif vertices == 5:
                label = "Pentagon"
            elif vertices == 6:
                label = "Hexagon"
            elif vertices == 7:
                label = "Heptagon"
            elif vertices == 8:
                label = "Octagon"
            elif vertices > 8:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (center, axes, angle) = ellipse
                    major_axis, minor_axis = max(axes), min(axes)
                    ellipse_ratio = minor_axis / float(major_axis)
                    if 0.5 < ellipse_ratio < 0.95:
                        label = "Ellipse"
                    elif circularity > 0.88:
                        label = "Circle"
                    else:
                        label = "Blob"
                except:
                    label = "Blob"
            else:
                label = "Blob"

            cv2.drawContours(labeled, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(labeled, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return Image.fromarray(labeled)

    def display_images(self):
        # Refresh the displayed images on the canvases
        self.tk_orig = ImageTk.PhotoImage(self.original_image)
        self.tk_proc = ImageTk.PhotoImage(self.processed_image)
        self.left_canvas.config(image=self.tk_orig)
        self.right_canvas.config(image=self.tk_proc)
        self.root.geometry(f"{self.original_image.width + self.processed_image.width + 100}x{max(self.original_image.height, self.processed_image.height) + 100}")

    def magnify_at_click(self, event):
        # Zoom into clicked region of processed image
        if self.processed_image is None:
            return

        x, y = event.x, event.y
        img_w, img_h = self.processed_image.size
        lbl_w, lbl_h = self.right_canvas.winfo_width(), self.right_canvas.winfo_height()
        scale_x = img_w / lbl_w
        scale_y = img_h / lbl_h
        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        region_size = 50
        left = max(0, img_x - region_size // 2)
        upper = max(0, img_y - region_size // 2)
        right = min(img_w, left + region_size)
        lower = min(img_h, upper + region_size)
        region = self.processed_image.crop((left, upper, right, lower))

        zoom_factor = 3
        region_np = np.array(region)
        zoomed_np = ndimage_zoom(region_np, (zoom_factor, zoom_factor, 1), order=3)
        zoomed_img = Image.fromarray(np.clip(zoomed_np, 0, 255).astype(np.uint8))

        popup = tk.Toplevel(self.root)
        popup.title(f"Zoom at ({img_x}, {img_y})")
        zoom_label = tk.Label(popup)
        zoom_label.pack()
        tk_zoom_img = ImageTk.PhotoImage(zoomed_img)
        zoom_label.config(image=tk_zoom_img)
        zoom_label.image = tk_zoom_img

    def check_camera(self):
        # Check if a webcam is available
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        return available

    def capture_image(self):
        # Capture an image from webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to access the camera.")
            return
        
        while True:
            ret, user_image = cap.read()
            
            if not ret:
                messagebox.showerror("Error", "Failed to capture an image.")
                break
            
            cv2.imshow("Press Space to Capture Image", user_image)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:
                cv2.imwrite("captured_image.jpg", user_image)
                messagebox.showinfo("Success", "Image captured successfully!")
                cv2.imshow("Captured Image", user_image)

                user_image_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
                pil_user_image = Image.fromarray(user_image_rgb)

                self.original_image = pil_user_image
                self.processed_image = pil_user_image.copy()

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                self.display_images()
                break 

        cap.release()

    def classify_image(self):
        # Use classifier to predict objects in the processed image
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to classify.")
            return

        # Convert processed PIL image to OpenCV BGR
        cv_img = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)

        # Temporarily save the processed image to pass to the classifier
        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, cv_img)

        # Run predictions (saves annotated_result.jpg with correct colors)
        self.classifier.predict_multiple_objects(temp_path)

        # Load and show the saved annotated image
        annotated_pil = Image.open("annotated_result.jpg").convert("RGB")
        self.processed_image = annotated_pil
        self.display_images()

    
    def load_model(self, model_path):
        # Load SVM model, PCA, and scaler from file
        with open(model_path, 'rb') as f:
            scaler, pca, model = pickle.load(f)
        self.classifier.scaler = scaler
        self.classifier.pca = pca
        return model, scaler


if __name__ == "__main__":
    root = tk.Tk()
    app = PSFViewer(root)
    root.mainloop()