# Vision-LSTM
> Outline a brief description of your project.

![Framework of Vision-LSTM](./img/Vision-LSTM.png)

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Results](#results)
* [Setup](#setup)
* [Usage](#usage)

## General Information
- Provide general information about your project here.
- What problem does it (intend to) solve?
- What is the purpose of your project?
- Why did you undertake it?
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Features
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3

## Results
In our urban village recognition case, the results can be seen in the following table.

| Method                                   | OA(%)     | Kappa     | F1        |
|------------------------------------------|-----------|-----------|-----------|
| No fusion (random image)                 | 88.1      | 0.634     | 0.708     |
| Average Pooling                          | 89.1      | 0.656     | 0.727     |
| Maximum Pooling                          | 79.3      | 0.461     | 0.588     |
| Element-wise Sum                         | 77.4      | 0.432     | 0.566     |
| **Vision-LSTM (proposed in this study)** | **91.6**  | **0.720** | **0.773** |

## Setup
What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located?

Proceed to describe how to install / setup one's local environment / get started with the project.

## Usage
How does one go about using it?
Provide various use cases and code examples here.

<!-- ## Citation -->
<!-- add bibtex here -->