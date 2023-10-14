from flask import Flask, render_template, request, redirect, url_for, flash
from models.model_detection import model_detection
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('home.html')
    
@app.route('/choose-action', methods=['POST'])
def choose_action():
    # Get the exercise choice from the form
    exercise_choice = request.form.get('exercise')
    button_pressed = request.form.get('submit_button')
    
    if button_pressed == "Upload":
        # Handle the file upload
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return render_template('result.html', exercise=exercise_choice, video_file=filename)
        else:
            flash('Invalid video format!')
            return redirect(url_for('index'))
    
    elif button_pressed == "Live":
        if exercise_choice == "Plank":
            model_detection(cap=1, model_path="./models/plank/LR_model.pkl", input_scaler_path="./models/plank/input_scaler.pkl")
        return redirect(url_for('index'))


@app.route('/clear-uploads-and-go-home')
def clear_uploads_and_go_home():
    # Delete all files inside 'uploads' folder
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
