#include "servo_controller.h"
#include <Arduino.h>
#include <ESP32Servo.h>

Servo servo;

ServoController::ServoController(int pin) {
	servoPin = pin;
	rotationSpeed = 0;
	isClockwise = true;
	isMoving = false;
	lastUpdateTime = 0;
	servo.attach(servoPin);
	servo.write(90);
}

void ServoController::setSpeed(int speed) {
	// Constrain speed between 0 and 1000
	rotationSpeed = constrain(speed, 0, 1000);

	// If speed is 0, just stop
	if(rotationSpeed == 0) stop();
	// Update servo position based on new speed
	else if(isMoving) update();
}

void ServoController::setDirection(bool clockwise) {
	isClockwise = clockwise;
	if(isMoving) update();
}

void ServoController::stop() {
	isMoving = false;
	servo.write(90);
}

void ServoController::rotate() {
	isMoving = true;
	lastUpdateTime = millis(); // Reset timing
	update();				   // Update immediately
}

bool ServoController::shouldMove(unsigned long currentTime) {
	if(rotationSpeed >= 1000) return true;
	if(rotationSpeed <= 0) return false;

	// Use modulo to create a position within the cycle (0-99)
	unsigned long positionInCycle = currentTime % cycleTime;

	// Calculate duty cycle threshold - how much of the cycle should be movement
	unsigned long threshold;

	threshold = map(rotationSpeed, 1, 999, 1, 100);
	// Determine if we should be moving at this moment
	return (positionInCycle < threshold);
}

void ServoController::update() {
	if(!isMoving) return;

	unsigned long currentTime = millis();

	// Determine if we should be moving based on duty cycle
	if(shouldMove(currentTime)) {
		int value = isClockwise ? 90 - servoActiveSpeed : 90 + servoActiveSpeed;
		servo.write(value);
	} else servo.write(90); // Stop during break interval

	lastUpdateTime = currentTime;
}