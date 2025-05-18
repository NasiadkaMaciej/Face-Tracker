#include "web_server.h"
#include "servo_controller.h"

WebServer ServoWebServer::server(80);
ServoController* ServoWebServer::servoController = nullptr;

ServoWebServer::ServoWebServer(ServoController* controller) { servoController = controller; }

void ServoWebServer::start() {
	server.on("/rotate", HTTP_GET, handleRotate);
	server.begin();
}

void ServoWebServer::handleClient() { server.handleClient(); }

void ServoWebServer::handleRotate() {
	String direction = server.arg("direction");
	int speed = server.arg("speed").toInt();

	if(direction == "left") servoController->setDirection(false);
	else servoController->setDirection(true);

	servoController->setSpeed(speed);
	servoController->rotate();

	if(speed == 0) {
		server.send(200, "text/plain", "Servo stopped\n");
		servoController->stop();
		return;
	}

	server.send(200, "text/plain", "Servo rotating " + direction + " at speed " + String(speed) + "\n");
}