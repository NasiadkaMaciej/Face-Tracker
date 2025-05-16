#pragma once

#include <WiFi.h>

class WiFiManager {
  public:
	void begin(const char* ssid, const char* password);
	void startAccessPoint(const char* ssid, const char* password);
	bool isConnected();
};