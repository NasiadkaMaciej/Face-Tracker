#pragma once

#include "servo_controller.h"
#include <WebServer.h>
#include <WiFi.h>

class ServoWebServer {
  public:
	ServoWebServer(ServoController* controller);
	void start();
	static void handleClient();

  private:
	static ServoController* servoController;
	static WebServer server;
	static void handleRotate();
	static void handleStop();
};