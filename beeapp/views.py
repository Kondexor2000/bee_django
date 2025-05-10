import base64
import os
import uuid
import cv2
from django.urls import reverse_lazy
import numpy as np
from django.shortcuts import get_object_or_404, render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.generic import CreateView, UpdateView, DeleteView
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.template.loader import get_template
from django.template import TemplateDoesNotExist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from keras.utils import to_categorical
from .models import Bee
from django.http import HttpResponse, Http404

# Helper function to check if template exists
def check_template(template_name, request):
    try:
        get_template(template_name)
    except TemplateDoesNotExist:
        messages.error(request, 'Brak pliku HTML')
        return False
    return True

class SignUpView(CreateView):
    form_class = UserCreationForm
    template_name = 'signup.html'
    success_url = reverse_lazy('login')

    def dispatch(self, request, *args, **kwargs):
        if not check_template(self.template_name, request):
            return HttpResponse("Template not found.")
        if request.user.is_authenticated:
            messages.info(request, "You are already registered and logged in.")
            return redirect('login')
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Registration successful. Please log in.")
        return response

class EditProfileView(LoginRequiredMixin, UpdateView):
    form_class = UserChangeForm
    template_name = 'edit_profile.html'
    success_url = reverse_lazy('login')

    def get_object(self):
        return self.request.user

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Profile updated successfully.")
        return response

class DeleteAccountView(LoginRequiredMixin, DeleteView):
    template_name = 'delete_account.html'
    success_url = reverse_lazy('login')

    def get_object(self, queryset=None):
        if self.request.user.is_authenticated:
            return self.request.user
        raise Http404("You are not logged in.")

    def delete(self, request, *args, **kwargs):
        try:
            response = super().delete(request, *args, **kwargs)
            messages.success(request, "Account deleted successfully.")
            return response
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('delete_account')

# Custom login view
class CustomLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

    def form_valid(self, form):
        if not check_template(self.template_name, self.request):
            return HttpResponse("Brak pliku .html")
        
        remember_me = form.cleaned_data.get('remember_me', False)
        if remember_me:
            self.request.session.set_expiry(1209600)
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('process_image')

# Custom logout view
class CustomLogoutView(LoginRequiredMixin, LogoutView):
    next_page = 'login'

# Process image view
@csrf_exempt
@login_required
def process_image(request):
    if not check_template('upload_image.html', request):
        return HttpResponse("Brak pliku .html")

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_name = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join('bee_images', image_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the file to the filesystem
        with open(image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Read the image for OpenCV processing
        image_file.seek(0)  # Reset the file pointer to the beginning
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.jpg', image)
        image_data = base64.b64encode(buffer).decode('utf-8')
        request.session['image_data'] = image_data

        # Save the image to the Bee model
        bee = Bee(user=request.user, image=image_name)
        bee.save()

        request.session['bee_id'] = bee.pk

        return redirect('detect_bee')  # Redirect to the result page after processing

    return render(request, 'upload_image.html')

# Detect bee view

@csrf_exempt
@login_required
def detect_bee(request):
    if not check_template('resulting.html', request):
        return HttpResponse("Brak pliku .html")

    if 'image_data' in request.session:
        image_data = base64.b64decode(request.session['image_data'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        bee_detected, predictions = process_bee_detection(image)

        # Get or create Bee object from database
        bee_id = request.session.get('bee_id')
        bee = get_object_or_404(Bee, id=bee_id)
        
        # Update is_bee field and save object
        bee.is_bee = bee_detected
        bee.save()

        # Store result in session
        request.session['bee_detected'] = bee_detected

        return redirect('display_results')
    return JsonResponse({'error': 'No image data found'}, status=400)

# Display results view
@login_required
def display_results(request):
    if not check_template('results.html', request):
        return HttpResponse("Brak pliku .html")

    bee_id = request.session.get('bee_id')
    if not bee_id:
        return HttpResponse("Brak danych dotyczących analizy")

    bee = get_object_or_404(Bee, id=bee_id)

    if bee.is_bee:
        report = 'Jest pszczoła. Możemy żyć szczęśliwie'
    else:
        report = 'Nie ma pszczół. Potrzebna pomoc'

    return render(request, 'results.html', {
        'report': report
    })

def process_bee_detection(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define yellow and black color ranges
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 30], dtype=np.uint8)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours for yellow and black areas
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
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(data, labels, epochs=10, batch_size=64, validation_split=0.2)
    predictions = model.predict(data)
    bee_detected = any(pred[1] > 0.5 for pred in predictions)

    return bee_detected, predictions.tolist()