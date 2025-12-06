# Fitness Trainer with Pose Estimation

An AI-powered web application that tracks exercises using computer vision and provides real-time feedback.

## Features

- Real-time pose estimation using MediaPipe
- Multiple exercise types: Squats, Push-ups, and Hammer Curls
- Customizable sets and repetitions
- Exercise form feedback
- Progress tracking
- Web interface for easy access
- Real time accuracy using trained models

## Usage

1. Start the Flask server:

   ```
   python app.py
   ```

2. Open a web browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

3. Select an exercise type, set your desired number of repetitions and sets, then click "Start Workout"

4. Position yourself in front of your camera so that your full body is visible

5. Follow the on-screen guidance to perform the exercise correctly

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and images
- `pose_estimation/` - Pose estimation modules
- `exercises/` - Exercise tracking classes
- `feedback/` - User feedback modules
- `utils/` - Helper functions and utilities
- `models/` - save all trained models and scalers
- `Datasets/` - datasets used to train the models

## Technologies Used

- Flask - Web framework
- OpenCV - Computer vision
- MediaPipe - Pose estimation
- HTML/CSS/JavaScript - Frontend
- Machine learning models like SVM, LR etc

## License

This project is licensed under the MIT License - see the LICENSE file for details.
