# SheetStack Counter: Web Application for Accurate Sheet Counting
Welcome to the SheetStack Counter application! This tool simplifies counting sheets in image stacks with intuitive image processing and visualization features.

Overview
This project provides a web application for counting the number of sheets in an image of sheet stacks using image processing techniques. The application is built with OpenCV for image processing and Streamlit for creating a user-friendly web interface.


Features:

* Upload an image of sheet stacks.
* Automatically count the number of sheets.
* View processed images with annotated results.

Important 

Before running the application, make sure you have the following installed:
Python 3.7 or higher
pip (Python package installer)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



Requirements:


step 1 : To run this application, you need to have the following libraries installed:

* opencv-python
* numpy
* streamlit
* Pillow
* Installation





step 2 : To Clone the repository:

    git clone <https://github.com/Rabbit3112/Counting-papers-using-OpenCV.git>


Now open the cloned directory and open your VS-Code or any other tool to run the code







Step 3: Create a virtual environment (optional but recommended):


    python -m venv venv
    source venv/bin/activate 






Step 4: Install the required packages:

Install the required libraries using pip by running 'requirements.txt' :


    pip install -r requirements.txt
    Running the Application
    Navigate to the project directory:







Step 5: Run the Streamlit Application:


    streamlit run app.py

This command will launch the web application and open it in your default web browser.






Step 6: Using the Application

* Upload an image

Click the "Choose an image..." button to upload an image of sheet stacks in JPEG, PNG, or JPEG format.



---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


View Results:

* The application will process the image and display the number of detected sheets.
* Intermediate images showing various processing stages will be displayed for debugging purposes.

Troubleshooting

* If the application does not start: Ensure you have activated the virtual environment and installed all dependencies.
* If you encounter any errors: Check the terminal output for error messages and ensure that your image is correctly formatted and of good quality.


Contributing

* Feel free to submit issues or pull requests if you have suggestions or improvements.


License
* This project is licensed under the MIT License - see the LICENSE file for details.
