<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/dechantoine/explainable-chess-engine">
    <img src="images/xplainable.png" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Explainable Chess Engine</h3>

  <p align="center">
    A minimalist DL chess engine with explainability.
    <br />
    <br />
    <a href="https://github.com/dechantoine/explainable-chess-engine">View Demo</a>
    ·
    <a href="https://github.com/dechantoine/explainable-chess-engine">Report Bug</a>
    ·
    <a href="https://github.com/dechantoine/explainable-chess-engine">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#objectives">Objectives</a></li>
        <li><a href="#roadmap">Roadmap</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT -->

## About The Project

This project aim to build a minimalist Deep Learning chess engine with explainability.
The goal is to create a chess engine that can explain its decisions in a human-readable way.
This project will be a great way to learn about chess engines, deep learning, and explainability in AI.

### Objectives

- Build a Deep Learning framework optimized for training with chess data.
- Monitor the performance of the training framework to optimize training time, CPU/GPU usage, and memory usage.
- Build the best model with the minimal number of parameters.
- Create a human-readable explanation of the model's decisions.
- Implement a chess engine that can play against human or AI players using the model.

### Roadmap

- [x] Create the project structure
- [x] Custom pytorch dataset for chess data
  - [x] Read any board state at any game number in a PGN file
  - [x] Convert board, moves and game result to tensor
  - [x] Batch all operations
  - [x] Tests
- [x] Training loop & utilities for training
  - [x] Training loop
  - [x] Logging & Tensorboard
  - [x] Evaluation
  - [x] Save & Load model
  - [ ] Tests for training framework
- [ ] Chess Engine
  - [x] Beam Search using the model
  - [ ] Evaluate against Stockfish
  - [ ] Tests for the chess engine
- [ ] Deployment
  - [ ] Dockerize the project
  - [ ] Deploy on Lichess
  - [ ] Deploy on HuggingFace

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

You need to have Python 3.11 installed on your machine.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/dechantoine/explainable-chess-engine.git
   ```
2. Install the project dependencies using Poetry.
   ```sh
   poetry install
   ```

### Run unit tests

```sh
poetry run python -m pytest
```

### Run performance tests

```sh
poetry run python -m profile_package.profile_dataset
```

### Train a model

```sh
poetry run python -m src.train.train
```

### Launch Tensorboard

```sh
poetry run tensorboard --logdir=runs
```

<!-- CONTACT -->

## Contact

[@dechantoine](https://twitter.com/AI_bIAses) - dechantoine@gmail.com
