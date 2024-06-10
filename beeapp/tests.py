import cv2
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Define a mock model for testing purposes
def create_mock_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 neurons for two classes (bee and non-bee)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Helper function to simulate image processing
def simulate_image_processing():
    # Load the image from the file
    image_path = 'tests/images/bee.jpg'  # Ensure the image is in this path
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not found at {image_path}')
    return image

class BeeDetectionTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.client.login(username='testuser', password='12345')
        self.mock_model = create_mock_model()

    def test_login_view(self):
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

    def test_logout_view(self):
        response = self.client.get(reverse('logout'))
        self.assertEqual(response.status_code, 302)  # Redirect to login page

    def test_upload_image_view(self):
        response = self.client.get(reverse('upload_image'))
        self.assertEqual(response.status_code, 200)

    def test_process_image_view(self):
        image = simulate_image_processing()
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        response = self.client.post(reverse('process_image'), {'image': image_bytes}, format='multipart')
        self.assertEqual(response.status_code, 302)  # Redirect to detect_bee

    def test_detect_bee_view(self):
        image = simulate_image_processing()
        _, buffer = cv2.imencode('.jpg', image)
        self.client.session['image_data'] = buffer.tobytes()
        response = self.client.get(reverse('detect_bee'))
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn('bee_detected', response_json)
        self.assertIn('predictions', response_json)

    def test_display_results_view(self):
        self.client.session['bee_detected'] = True
        self.client.session['predictions'] = [[0.1, 0.9]]
        response = self.client.get(reverse('display_results'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Proponujemy bardziej oszczędne światło')

    def test_process_bee_detection_function(self):
        image = simulate_image_processing()
        bee_detected, predictions = process_bee_detection(image)
        self.assertIsInstance(bee_detected, bool)
        self.assertIsInstance(predictions, list)

    def tearDown(self):
        self.user.delete()

# Mocking the process_bee_detection function for tests
def process_bee_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 30], dtype=np.uint8)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    labels = []

    for contour_yellow in contours_yellow:
        for contour_black in contours_black:
            x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(contour_yellow)
            x_black, y_black, w_black, h_black = cv2.boundingRect(contour_black)
            if x_yellow < x_black + w_black and x_yellow + w_yellow > x_black and y_yellow < y_black + h_black and y_yellow + h_yellow > y_black:
                bee_image = image[y_yellow:y_yellow + h_yellow, x_yellow:x_yellow + w_yellow]
                resized_bee_image = cv2.resize(bee_image, (32, 32))
                data.append(resized_bee_image)
                labels.append(1)

    data = np.array(data).astype('float32') / 255.0
    labels = np.array(labels)
    
    if data.size == 0:
        return False, []

    # One-hot encoding of labels
    labels = to_categorical(labels, 2)

    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 neurons for two classes (bee and non-bee)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(data, labels, epochs=10, batch_size=64, validation_split=0.2)
    predictions = model.predict(data)
    bee_detected = any(pred[1] > 0.5 for pred in predictions)

    return bee_detected, predictions.tolist()