# SheetStack Counter: Web Application for Accurate Sheet Counting
Overview
This project provides a web application for counting the number of sheets in an image of sheet stacks using image processing techniques. The application is built with OpenCV for image processing and Streamlit for creating a user-friendly web interface.

Features
Upload an image of sheet stacks.
Automatically count the number of sheets.
View processed images with annotated results.
Requirements
To run this application, you need to have the following libraries installed:

opencv-python
numpy
streamlit
Pillow
Installation
Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Running the Application
Navigate to the project directory:

bash
Copy code
cd <repository-directory>
Start the Streamlit server:

bash
Copy code
streamlit run app.py
This command will launch the web application and open it in your default web browser.

Using the Application
Upload an image:

Click the "Choose an image..." button to upload an image of sheet stacks in JPEG, PNG, or JPEG format.
View Results:

The application will process the image and display the number of detected sheets.
Intermediate images showing various processing stages will be displayed for debugging purposes.
Troubleshooting
If the application does not start: Ensure you have activated the virtual environment and installed all dependencies.
If you encounter any errors: Check the terminal output for error messages and ensure that your image is correctly formatted and of good quality.
Contributing
Feel free to submit issues or pull requests if you have suggestions or improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.
