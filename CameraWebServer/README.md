This folder contains the Arduino code for the ESP32-Cam, which is responsible for control of the 12V Solenoid lock via Relay Module.

For the main function of receiving the HTTP request and its relevant endpoint, check around line 1147 in app_httpd.cpp for more details.

The main .ino file mostly just contains initialzation code for the default web camera server and the AsyncWebServer used to receive
HTTP requests from the Python Script.

Required libraries are as follows:<br/>
ESPAsyncWebServer <br/>
AsyncTCP <br/>
ESP32 Libraries <br/>
