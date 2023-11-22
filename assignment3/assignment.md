# How to run the code

The code can be run either via Docker or via PIP.

1. Docker

With Docker installed, typing "docker compose up" in terminal will run the code.

2. PIP

First, install the libraries by typing "pip install -r requirements.txt" to the terminal.
Then, typing "python assignment.py" in the assignment3 directory will run the code.

The assignment results will be printed out to the console.

# Sequential Recommendations Method

The method proposed for generating sequential group recommendations leverages a diverse approach in generating movie recommendations for a user group. It utilizes a sequential process, employing a separate kNN model for each recommendation sequence. Within each sequence, the model dynamically selects varying neighboring users and explores their unrated movies, opting for random selections instead of consistently favoring top-rated choices. This multi-stage diversity augmentation results in distinct sets of movie suggestions across the three sequences, ensuring varied and personalized recommendations for the user group to explore together.
