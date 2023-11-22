# How to run the code

The code can be run either via Docker or via PIP.

1. Docker

With Docker installed, typing "docker compose up" in terminal will run the code.

2. PIP

First, install the libraries by typing "pip install -r requirements.txt" to the terminal.
Then, typing "python assignment.py" in the assignment3 directory will run the code.

The assignment results will be printed out to the console.

# The method

The method proposed for generating sequential group recommendations leverages collaborative filtering and a k-nearest neighbors to find similar users within a group based on movie preferences. By aggregating recommendations from diverse neighbors and introducing randomness, it suggests sequences of top-rated movies for the entire group. This approach balances individual preferences and collective appeal, aiming to offer varied movie sets catering to different tastes while ensuring a shared viewing experience.
