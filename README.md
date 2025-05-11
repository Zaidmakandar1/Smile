# Smile Detection Web Application

This project is a web-based smile detection application built using Python's Flask framework and OpenCV. It uses your webcam to detect a smile in real-time, and once a smile is detected, it automatically takes a picture and displays it on the screen.

## Features

- **Real-Time Smile Detection**: Uses the webcam to detect smiles in real time.
- **Automatic Photo Capture**: Once a smile is detected, a photo is automatically taken and displayed on the screen.
- **Web Interface**: The entire process takes place through the browser without the need for additional software.

## Installation

1. Clone the Repository:

   ```bash
   git clone https://github.com/Shahid6174/Smile.git
   cd Smile
   ```

2. Create a Virtual Environment (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Application:

   ```bash
   python app.py
   ```

2. Access the Web Interface:

   Open your browser and go to:  
   http://127.0.0.1:5000/

3. Smile for the Webcam:

   The webcam will start, and once a smile is detected, a photo will be automatically taken and displayed on the screen.

## Project Structure

Smile/  
├── app.py  
├── requirements.txt  
├── templates/  
│ └── index.html  
├── static/  
│ ├── uploads/  
│ └── images/  
│  
└── dataset/  
 └── haarcascade_smile.xml

- **app.py**: Main Flask application.
- **requirements.txt**: Python dependencies.
- **templates/index.html**: HTML template.
- **static/images/**: Stores clicked images.
- **dataset/haarcascade_smile.xml**: Haar cascade for smile detection.

## Dependencies

- Flask
- OpenCV (cv2)
- Werkzeug

_All dependencies are listed in the requirements.txt file._

## License

This project is licensed under the MIT License.

## Acknowledgements

- OpenCV for the smile detection classifier
- Flask for the web framework

## Contributing

We welcome contributions to this project! To get started:

1. **Fork the repository**: Click the "Fork" button at the top-right of this page.
2. **Clone the repository**:  
   After forking, clone the repository to your local machine using:
   ```bash
   git clone https://github.com/your-username/Smile.git
   ```
3. **Create a new branch**:  
   Always create a new branch for your feature or fix:
   ```bash
   git checkout -b your-branch-name
   ```
4. **Make your changes**:  
   Modify the code, and make sure to test your changes.
5. **Commit your changes**:  
   Commit your changes with a clear message:
   ```bash
   git commit -m "Describe your changes"
   ```
6. **Push to your fork**:  
   Push the changes to your forked repository:
   ```bash
   git push origin your-branch-name
   ```
7. **Open a pull request**:  
   Go to the original repository and click on "New Pull Request."

Please make sure your code adheres to the existing style and is well-tested.
