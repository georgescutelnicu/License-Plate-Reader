### License Plate Reader
Identifies the license plate, cuts and displays it, and extracts the text from it. <br>
An image with higher resolution and clearer license plate will have a better accuracy.<br>
[The model was trained on european car plates]

<img src="extras/header.jpg" width="400">

### Requirements:
* [NumPy](http://www.numpy.org/)
* [OpenCV](https://docs.opencv.org/4.x/)
* [Matplotlib](http://matplotlib.org/)
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

### Example
Input:
<br>
<img src="extras/car.jpg" width="300" height="350">

Output:
<br>
<img src="extras/plate.png" width="300">

Confidence Score:
<br>
<img src="extras/confidence.png" width="600">

### Demo
<a href="https://huggingface.co/spaces/georgescutelnicu/license-plate-reader">
    <img src="https://img.shields.io/badge/Deployed%20on%20Hugging%20Face%20with%20Gradio-FFA500"></img>
</a>
