#pragma once

#include <Arduino.h>

class ServoController {
  public:
	ServoController(int pin);
	void setSpeed(int speed); // Values 0-1000
	void setDirection(bool clockwise);
	void stop();
	void rotate();
	void update();

  private:
	int servoPin;
	int rotationSpeed; // 0-1000
	bool isClockwise;
	bool isMoving;

	// Timing variables
	unsigned long lastUpdateTime;
	const int servoActiveSpeed = 10; // Fixed servo speed when moving
	const unsigned long cycleTime = 100;

	// Calculate if we should be moving at the current time based on duty cycle
	bool shouldMove(unsigned long currentTime);
};