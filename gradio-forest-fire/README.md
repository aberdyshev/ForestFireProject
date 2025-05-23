# Gradio Forest Fire Animation Project

This project simulates a forest fire using a cellular automaton model and provides an interactive interface using Gradio. Users can visualize the spread of fire in a forest based on various parameters.

## Project Structure

```
gradio-forest-fire
├── src
│   └── app.py          # Main application code for the Gradio forest fire animation
├── requirements.txt     # Python dependencies required for the project
├── Dockerfile           # Instructions to build a Docker image for the project
└── README.md            # Documentation for the project
```

## Installation

To run this project, you need to have Python installed along with the required dependencies. You can install the dependencies using the following command:

```
pip install -r requirements.txt
```

## Running the Application

To run the application, execute the following command:

```
python src/app.py
```

This will start a local server, and you can access the Gradio interface in your web browser.

## Docker

To build and run the Docker container for this project, use the following commands:

1. Build the Docker image:

   ```
   docker build -t gradio-forest-fire .
   ```

2. Run the Docker container:

   ```
   docker run -p 7860:7860 gradio-forest-fire
   ```

You can then access the application at `http://localhost:7860`.

## Parameters

The application allows you to adjust various parameters for the forest fire simulation:

- **Random Seed**: Seed for random number generation.
- **Grid Size**: Size of the grid for the simulation.
- **Forest Size**: Dimensions of the forest (NxN).
- **Number of Frames**: Total frames for the animation.
- **Tree Growth Probability (p)**: Probability of tree growth.
- **Lightning Probability (f)**: Probability of lightning strikes.
- **Initial Forest Fraction**: Fraction of the grid that is initially forested.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.