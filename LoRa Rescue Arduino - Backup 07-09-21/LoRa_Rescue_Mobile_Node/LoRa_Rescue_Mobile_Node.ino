// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

// Defines LED pin
#define led 6

byte destination = 0xAA;      // destination: Gateway broadcast

String phoneNum = "";
int i=0;

void setup() {
  // Starts UART connection in Serial1
  // Enable pins
  pinMode(led,OUTPUT);
  Serial.begin(115200);
  Serial1.begin(9600);
  
  // Starts LoRa and blinks when it fails to connect
  if (!LoRa.begin(868E6)) {
    Serial.println("Starting LoRa failed!");
    while (1){
      delay(2000);
      digitalWrite(led,HIGH);
      delay(2000);
      digitalWrite(led,LOW);
    }
  }
  
  // Sets LoRa preferences
  LoRa.disableInvertIQ();
  LoRa.setTxPower(18.5,PA_OUTPUT_PA_BOOST_PIN);
  LoRa.setSignalBandwidth(125E3);
  LoRa.setSpreadingFactor(9);
  LoRa.enableCrc();

  // Debugging
  delay(1000);
  Serial.println("Device Started!");
  digitalWrite(led, HIGH);
  delay(2000);
}

void loop() {
  // Checks if there is UART data from the ESP32
  digitalWrite(led, LOW);
  if(Serial1.available()){
    phoneNum = Serial1.readString();
    if(phoneNum == "Server started"){
      phoneNum = "";
    } else {
      Serial.print("Phone: ");
      Serial.print(phoneNum);
    }
  }
  
  // Only sends LoRa once phone number is received
  if(phoneNum != ""){
    digitalWrite(led,HIGH);
    
    ///// Insert LoRa send code here
    
    for(int i=0; i<60; i++){
      // send packet
      LoRa.beginPacket();
      if(i == 0){
        LoRa.write(destination); // destination
      }
      LoRa.print(phoneNum);
      LoRa.endPacket();
      delay(100);
    }
    
    ///// End of LoRa send code
    
    // Reset phoneNum to "" to avoid sending unwanted signals
    i = 0;
    phoneNum = "";
  }
}
