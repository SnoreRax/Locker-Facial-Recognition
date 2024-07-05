The following is our final project for Algorithms and Complexities and Architecture and Organization.
<br/>
The members included are as follows:<br/>
Alboroto, Rojer Angelo <br/>
Rjindorp, Nathaniel <br/>
Sian, David Anthony <br/>
<br/>
The Archi---Algo-Final-Product-main folder contains our Python code and the CameraWebServer Folder contains our Arduino code.

The Python code is responsible for running the required Flask server and for the use of the FisherFace Algorithm, while
the Arduino code is responsible for running the ESP32-Cam, using its built-in camera, running its default web server to use
the camera, and running another AsyncWebServer with the sole purpose of receiving HTTP requests from the Flask server once
an individual's face has been detected.

For full context of the project, the full title of the project is Entry Sentry: A Door Lock System For High School Lockers 
Using ESP32 Camera and Fisherface Algorithm. The concept of the project is to develop a facial recognition-based locker system,
which uses the FisherFace Algorithm for processing facial input, and the ESP32-Cam to detect facial input and as the main controller
of the lock system itself. The control of the lock is to be done via a relay module connected to the ESP32-Cam, and only activated after
a signal is sent from the ESP32-Cam. As for the ESP32-Cam, it will only send a signal after it has received an HTTP request from the Flask
server to unlock the solenoid used, by which it will only send an HTTP request after it determines that the facial image received is the 
owner of the locker.

The materials used were as follows: ESP32-Cam, Relay Module (Used 2 channel, needs only 1 channel), 12V Solenoid Lock, 12V AC/DC Power
Adapter, DC Barrel Socket, Breadboard, male-to-male wires, male-to-female wires. Container used was a makeshift cardboard box that
has a transparent top.
