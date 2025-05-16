#include "wifi_manager.h"
#include <WiFi.h>

const char* ap_ssid = "ESP32_Servo_Controller";
const char* ap_password = "your_password_here";

void WiFiManager::begin(const char* ssid, const char* password) {
	WiFi.begin(ssid, password);
	while(WiFi.status() != WL_CONNECTED)
		delay(1000);
}

void WiFiManager::startAccessPoint(const char* ssid, const char* password) { WiFi.softAP(ssid, password); }

bool WiFiManager::isConnected() { return WiFi.status() == WL_CONNECTED; }