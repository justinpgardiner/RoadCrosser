# RoadCrosser
This project was completed as a part of my IB Diploma Extended Essay in Computer Science.
## Preview
Game developed in Python using the Pygame library
The game is based off of a game such as Crossy Road, or Frogger, where the objective is to cross the road for as long as possible.

<img width="762" alt="Screenshot 2024-09-07 at 7 47 05 PM" src="https://github.com/user-attachments/assets/e6806868-3524-498b-b67d-ca3920f2a392">

From the menu screen, the user can choose to either play the game themselves, or watch trained artificial intelligence models attempt to complete the game.
## Models
Multiple AI models were implemented and trained to complete this game.
### NeuroEvolution of Augmenting Topologies (NEAT)
The NEAT algorithm is implemented using the library neat-python, and is sufficiently successful at completing the game.
Success could be improved by improving configuration settings for the model and incentivizing motion the left and right, instead of only forward.
### Basic Artificial Neural Network
The basic artificial neural network implemented to solve this game was implemented using Pytorch. The model is simple to the point of uselessness, and can be improved upon in the future by adding more complexity.
