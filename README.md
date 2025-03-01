# Refactored Adventure

## Overview
The Refactored Adventure project is designed to fine-tune and utilize a Stable Diffusion model with ControlNet for image generation tasks. This project includes scripts for fine-tuning the model, running inference, and providing a user-friendly web interface for generating images based on user inputs.

## Project Structure
```
refactored-adventure
├── run.sh          # Shell script to launch the fine-tuning process
├── finetune.py     # Script for fine-tuning the Stable Diffusion model
├── app.py          # Gradio web interface for image generation
├── inference.py     # Script for running inference with the fine-tuned model
└── README.md       # Documentation for the project
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd refactored-adventure
   ```

2. **Install required packages**:
   Ensure you have Python and pip installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare your images**:
   Place your training images in a directory and update the `run.sh` script with the correct path.

## Usage

### Fine-tuning the Model
To fine-tune the Stable Diffusion model, run the following command:
```
bash run.sh
```
This will start the fine-tuning process with the specified parameters in the `run.sh` script.

### Generating Images with the Web Interface
To launch the Gradio web interface, run:
```
python app.py
```
You can input prompts, upload initial images, and adjust parameters like strength and inference steps to generate new images.

### Running Inference
To run inference with the fine-tuned model, use the following command:
```
python inference.py --model_path <path-to-fine-tuned-model> --init_image <path-to-initial-image> --prompt "<your-prompt>" --strength <value> --num_inference_steps <number> --out_image <output-image-path>
```
Replace the placeholders with your specific values.

## Example
1. Fine-tune the model using your dataset.
2. Launch the Gradio app to generate images interactively.
3. Use the inference script for batch processing or automated tasks.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.