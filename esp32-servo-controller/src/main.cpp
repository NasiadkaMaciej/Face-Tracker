#include "servo_controller.h"
#include "web_server.h"
#include "wifi_manager.h"
#include <WiFi.h>

const char* ssid = "ESP32_Servo_Controller";
const char* password = "password";
WiFiManager wifiManager;
ServoController* servoController;
ServoWebServer* webServer;

void setup() {
	Serial.begin(115200);
	wifiManager.startAccessPoint(ssid, password);
	servoController = new ServoController(13);
	webServer = new ServoWebServer(servoController);
	webServer->start();
}

void loop() {
	ServoWebServer::handleClient();
	servoController->update();
}