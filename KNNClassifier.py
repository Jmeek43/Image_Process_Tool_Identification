import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import pickle

# Main classifier class for tool recognition
class KNNClassifier:
    def __init__(self):
        # Initialize storage for features and labels
        self.X = []
        self.y = []

        # Toggle switches for feature usage
        self.use_canny = True
        self.use_critical = True
        self.use_shape = True
        self.use_hog = True
        self.use_lbp = True
        self.use_gradient = True

        # Machine learning model components
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None

        # Dataset folder path
        self.dataset_folder = os.getcwd() + "/dataset/"

        # Feature sizes for different types of extracted features
        self.canny_feature_size = 256
        self.critical_feature_size = 8
        self.shape_feature_size = 40
        self.hog_feature_size = 128
        self.lbp_feature_size = 10
        self.intensity_feature_size = 32
        self.gradient_feature_size = 10

    def calculate_additional_shape_features(self, contour, normalize_hu=True):
        # Calculates advanced shape features such as solidity, aspect ratio, extent, orientation, and Hu moments
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        if normalize_hu:
            hu_moments = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu_moments]

        hull = cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h + 1e-6)

        # Use PCA to find main orientation of shape
        contour_array = contour.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(contour_array, mean=None)[:2]
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
        angle = (angle + np.pi) / (2 * np.pi)

        return [solidity, aspect_ratio, extent, angle] + hu_moments

    def preprocess_background(self, image):
        # Remove bright background by thresholding and masking
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(image, image, mask=mask)

    def augment_image(self, image):
        # Augment image by small rotations
        angles = [-10, -5, 5, 10]
        aug_images = [image]
        for angle in angles:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            aug_images.append(rotated)
        return aug_images

    def calculate_curvature(self, contour):
        # Calculate curvature statistics from contour points
        if len(contour) < 5:
            return 0.0, 0.0, 0.0
        contour = contour.squeeze()
        dx = np.gradient(contour[:, 0])
        dy = np.gradient(contour[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(ddx * dy - dx * ddy) / (dx * dx + dy * dy + 1e-8) ** 1.5
        return np.mean(curvature), np.std(curvature), np.max(curvature)

    def extract_lbp_features(self, image):
        # Extract Local Binary Pattern features for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_feature_size, range=(0, self.lbp_feature_size))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        return hist

    def critical_points(self, gray, num_points=200, radius=5):
        # Detect critical points using second derivatives
        gray_blur = gaussian_filter(gray, sigma=1)
        Ixx = gaussian_filter(gray_blur, sigma=1, order=(2, 0))
        Iyy = gaussian_filter(gray_blur, sigma=1, order=(0, 2))
        Ixy = gaussian_filter(gray_blur, sigma=1, order=(1, 1))
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        R = det - 0.04 * (trace ** 2)
        R_norm = (R - R.min()) / (R.max() - R.min() + 1e-5)
        flat_indices = np.argsort(R_norm.ravel())[::-1]
        h, w = R.shape
        keypoints = [(i % w, i // w) for i in flat_indices[:num_points]]
        return keypoints

    def extract_critical_points(self, image, num_points=200, radius=5):
        # Normalize critical points for robustness
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        keypoints = self.critical_points(img_gray, num_points=num_points, radius=radius)
        normed = np.array([[x / w, y / h] for (x, y) in keypoints])
        if len(normed) > 0:
            stats = np.hstack([
                np.mean(normed, axis=0),
                np.std(normed, axis=0),
                np.min(normed, axis=0),
                np.max(normed, axis=0)
            ])
        else:
            stats = np.zeros(8)
        return stats
    

    def extract_hog_features(self, image):
        # Extract Histogram of Oriented Gradients (HOG) features
        image = self.preprocess_background(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        features = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        return features[:self.hog_feature_size] if len(features) >= self.hog_feature_size else np.pad(features, (0, self.hog_feature_size - len(features)))

    def extract_canny_edges(self, image):
        # Extract Canny edge features
        image = self.preprocess_background(image)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return cv2.resize(edges, (32, 32)).flatten()

    def extract_gradient_features(self, image: np.ndarray) -> np.ndarray:
        # Extract gradient magnitude features using Sobel operator
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        hist, _ = np.histogram(grad_mag.ravel(), bins=self.gradient_feature_size, range=(0, 255))
        return hist / (hist.sum() + 1e-6)

    def extract_features_from_cropped(self, cropped_image):
        # Extract and concatenate all selected features from a cropped image
        feature_parts = []

        if self.use_canny:
            canny_edges = self.extract_canny_edges(cropped_image)
            feature_parts.append(self.pad_or_truncate(canny_edges, self.canny_feature_size))

        if self.use_critical:
            critical_points = self.extract_critical_points(cropped_image)
            feature_parts.append(critical_points)

        if self.use_shape:
            shape_features = self.extract_shapes(cropped_image)
            feature_parts.append(shape_features)

        if self.use_hog:
            hog_features = self.extract_hog_features(cropped_image)
            feature_parts.append(hog_features)

        if self.use_lbp:
            lbp_features = self.extract_lbp_features(cropped_image)
            feature_parts.append(lbp_features)

        if self.use_gradient:
            grad = self.extract_gradient_features(cropped_image)
            feature_parts.append(self.pad_or_truncate(grad, self.gradient_feature_size))

        intensity_features = self.extract_intensity_profile(cropped_image)
        feature_parts.append(intensity_features)

        if not feature_parts:
            print("Warning: No feature types enabled for cropped image.")
            return np.zeros(self.scaler.n_features_in_)

        combined_features = np.concatenate(feature_parts, axis=0)
        return combined_features

    def prepare_data(self):
        # Load dataset images, apply augmentation, and extract features
        from glob import glob
        image_paths = []
        labels = []

        for folder in os.listdir(self.dataset_folder):
            class_path = os.path.join(self.dataset_folder, folder)
            if os.path.isdir(class_path):
                for img_file in glob(f"{class_path}/*.*"):
                    image_paths.append(img_file)
                    labels.append(folder)

        for image_path, label in zip(image_paths, labels):
            original = cv2.imread(image_path)
            augmented_images = self.augment_image(original)
            for img in augmented_images:
                features = self.extract_features_from_cropped(img)
                self.X.append(features)
                self.y.append(label)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X = self.scaler.fit_transform(self.X)

    def extract_shapes(self, image):
        # Extract shape-based features from an image
        image = self.preprocess_background(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=60, sigma_r=0.4)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(self.shape_feature_size)

        cnt = max(contours, key=cv2.contourArea)
        basic_features = self.calculate_additional_shape_features(cnt, normalize_hu=True)
        return self.pad_or_truncate(np.array(basic_features), self.shape_feature_size)

    def extract_intensity_profile(self, image):
        # Extract intensity histogram from grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [self.intensity_feature_size], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-6)
        return hist

    def log_prediction_confidence(self, features):
        # Log prediction probability estimates
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba([features])
            print("Class probabilities:", probs)

    def pad_or_truncate(self, feature_array, target_size):
        # Adjust feature array to match target size
        if len(feature_array) > target_size:
            return feature_array[:target_size]
        else:
            return np.pad(feature_array, (0, target_size - len(feature_array)), mode='constant')

    def label_images(self):
        # Label all images in the dataset by folder name
        image_paths = []
        labels = []
        for folder_name in os.listdir(self.dataset_folder):
            folder_path = os.path.join(self.dataset_folder, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(folder_path, image_name))
                        labels.append(folder_name)
        return image_paths, labels

    def train_model(self):
        # Train an ensemble voting classifier using extracted features
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        self.prepare_data()
        self.pca = PCA(n_components=0.95)
        X_reduced = self.pca.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, self.y, test_size=0.2, random_state=42)

        svc = SVC(C=1, gamma='scale', kernel='rbf', probability=True)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)

        self.model = VotingClassifier(estimators=[
            ('svc', svc),
            ('rf', rf),
            ('knn', knn)
        ], voting='soft')

        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy (Ensemble + PCA): {accuracy * 100:.2f}%")

        with open('svm_pca_model.pkl', 'wb') as model_file:
            pickle.dump((self.scaler, self.pca, self.model), model_file)
        print("Scaler, PCA, and ensemble model saved.")

    def predict_multiple_objects(self, image_path):
        # Predict tool classes for multiple detected objects in an image
        image = cv2.imread(image_path)
        bboxes = self.detect_objects(image)

        class_colors = {
            'wrench': (26, 132, 18),
            'screwdriver': (255, 0, 0),
            'hammer': (0, 0, 255),
            'pliers': (0, 0, 0),
            'drill': (242, 74, 130),
        }

        predictions = []
        for bbox in bboxes:
            x, y, w, h = bbox
            cropped_image = image[y:y + h, x:x + w]

            features = self.extract_features_from_cropped(cropped_image)
            features = self.pad_or_truncate(features, self.scaler.n_features_in_)
            features_scaled = self.scaler.transform([features])
            features_pca = self.pca.transform(features_scaled)
            prediction = self.model.predict(features_pca)[0]

            color = class_colors.get(prediction.strip().lower(), (104, 88, 43))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, prediction, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            predictions.append((bbox, prediction))

        result_path = "annotated_result.jpg"
        cv2.imwrite(result_path, image)

        return predictions

    def detect_objects(self, image, min_area=500, iou_threshold=0.3):
        # Detect objects (tools) by thresholding and contour analysis
        debug_overlay = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=60, sigma_r=0.4)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((40, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]

        boxes = []
        rejected_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                rejected_boxes.append((x, y, w, h))
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 0.03 * image.shape[0] * image.shape[1]:
                rejected_boxes.append((x, y, w, h))
                continue
            aspect = w / float(h)
            if aspect < 0.05 or aspect > 8.0:
                rejected_boxes.append((x, y, w, h))
                continue
            boxes.append([x, y, x + w, y + h])

        # Apply Non-Maximum Suppression to filter overlapping boxes
        boxes = np.array(boxes)
        if len(boxes) == 0:
            return []

        keep = []
        idxs = np.argsort([-(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes])

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(boxes[i])
            iou_scores = []
            for j in idxs[1:]:
                xx1 = max(boxes[i][0], boxes[j][0])
                yy1 = max(boxes[i][1], boxes[j][1])
                xx2 = min(boxes[i][2], boxes[j][2])
                yy2 = min(boxes[i][3], boxes[j][3])
                inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                union_area = area_i + area_j - inter_area
                iou = inter_area / (union_area + 1e-6)
                iou_scores.append(iou)
            idxs = idxs[1:][np.array(iou_scores) < iou_threshold]

        final_boxes = []
        for i, box in enumerate(keep):
            x1, y1, x2, y2 = box
            is_inside = False
            for j, other in enumerate(keep):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2 = other
                if x1 >= ox1 and y1 >= oy1 and x2 <= ox2 and y2 <= oy2:
                    is_inside = True
                    break
            if not is_inside:
                final_boxes.append(box)

        # Draw rejected boxes for debugging
        for (x, y, w, h) in rejected_boxes:
            cv2.rectangle(debug_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite("debug_rejected_boxes.jpg", debug_overlay)

        return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in final_boxes]

    def plot_confusion_matrix(self, X_test, y_test):
        # Plot confusion matrix for evaluation
        ConfusionMatrixDisplay.from_estimator(self.model, X_test, y_test)
        plt.title("Tool Classification Confusion Matrix")
        plt.show()

# Main script execution
if __name__ == "__main__":
    classifier = KNNClassifier()
    classifier.train_model()

    new_image_path = "example.jpg"
    predictions = classifier.predict_multiple_objects(new_image_path)

    print(f"Predictions for {new_image_path}:")
    for bbox, predicted_class in predictions:
        print(f"Bounding Box: {bbox}, Predicted Class: {predicted_class}")