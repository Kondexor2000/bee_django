from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from .models import Bee
import cv2
from django.core.files.uploadedfile import SimpleUploadedFile

class BeeDetectionIntegrationTest(TestCase):

    def setUp(self):
        # Create a test user
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client = Client()
        self.client.login(username='testuser', password='testpass')

    
    def create_test_image(self):
        # Load an image from file for testing
        image_path = 'C:/Users/kondz/OneDrive/Pulpit/nlp/bee/Pszczola.jpg'
        image = cv2.imread(image_path)
        
        # Encode the image to JPEG format
        _, buffer = cv2.imencode('.jpg', image)
        image_data = buffer.tobytes()

        return SimpleUploadedFile('Pszczola.jpg', image_data, content_type='image/jpeg')

    def test_process_image_view(self):
        image = self.create_test_image()
        response = self.client.post(reverse('process_image'), {'image': image}, format='multipart')
        
        # Check for 302 redirect
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('detect_bee'))
        
        # Follow the redirect and check the response
        response = self.client.get(reverse('detect_bee'))
        self.assertEqual(response.status_code, 302)  # Make sure it redirects to display_results

        # Optionally, follow the next redirect to display_results
        response = self.client.get(response.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'results.html')

        # Add more assertions based on the expected behavior of display_results view
        self.assertContains(response, 'Jest pszczoła.')  # Example content check

        # You can also check session variables if necessary
        bee_detected = self.client.session.get('bee_detected')
        self.assertTrue(bee_detected)

    def test_detect_bee_view(self):
        image = self.create_test_image()
        response = self.client.post(reverse('process_image'), {'image': image}, format='multipart')
        self.assertEqual(response.status_code, 302)

        response = self.client.get(reverse('detect_bee'))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('display_results'))

        bee_detected = self.client.session.get('bee_detected')
        self.assertIsNotNone(bee_detected)

    def test_display_results_view(self):
        image = self.create_test_image()
        response = self.client.post(reverse('process_image'), {'image': image}, format='multipart')
        self.assertEqual(response.status_code, 302)
        
        response = self.client.get(reverse('detect_bee'))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('display_results'))

        response = self.client.get(reverse('display_results'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'results.html')

        bee_detected = self.client.session.get('bee_detected')
        report = 'Jest pszczoła. Możemy żyć szczęśliwie' if bee_detected else 'Nie ma pszczół. Potrzebna pomoc'
        self.assertContains(response, report)

    def tearDown(self):
        # Clean up any created data
        Bee.objects.all().delete()
        self.client.logout()

if __name__ == '__main__':
    TestCase.main()