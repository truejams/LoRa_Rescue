// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

// Defines LED pin
#define led 6

byte destination = 0xAA;      // destination: Gateway broadcast
String phoneNum = "";
int i=0;

// del after test
char isPhone;
#define reset 1
byte subNode = 0xCC; 
byte subNodeHop = 0xBA;
byte doneSubGnodeB = 0xAB;
int counter = 0;
int timer = 0;
int dataRec = 0;
char phone[10];
int rxRSSI[60];
int source;
char buf [4];
int dataSize = 60;
unsigned long currentTime;
unsigned long delayTime;
String RssiReading;

void setup() {
  // Starts UART connection in Serial1
  // Enable pins
  pinMode(led,OUTPUT);
  Serial.begin(115200);
  Serial1.begin(9600);

  // Delete after test
  memset (rxRSSI, 0, dataSize);
  memset (phone, 0, sizeof(phone));
  digitalWrite(reset, HIGH);
  pinMode(reset,OUTPUT);
  
  // Starts LoRa and blinks when it fails to connect
  if (!LoRa.begin(868E6)) {
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

  // Debugging
  delay(1000);
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
      ///// Insert LoRa send code here
      digitalWrite(led,HIGH);
      for(int i=0; i<60-1; i++){
        // send packet
        LoRa.beginPacket();
        LoRa.write(destination); // destination
        LoRa.print(phoneNum);
        LoRa.endPacket();
        delay(100);
      ///// End of LoRa send code
      }
      digitalWrite(led, LOW);
      LoRa.receive();
      rxMode();
    }
    // Reset phoneNum to "" to avoid sending unwanted signals
    i = 0;
    phoneNum = "";
  }
  
  // Reset phoneNum to "" to avoid sending unwanted signals
  i = 0;
  phoneNum = "";
}

void rxMode(){
  counter = 0;
  while(1){
    currentTime = millis();
    int packetSize = LoRa.parsePacket();
    if (packetSize) {
      // received a packet
      source = LoRa.read();
      if (source == subNodeHop && counter <= dataSize) {
        dataRec = 1;
        if (counter == 0) Serial.println("Receiving C"); 
        digitalWrite(led,HIGH);
        counter++;
        // Reads packet
        int i = 0;
        while (LoRa.available()){
          if(counter == 1){
            phone[i] = (char)LoRa.read();
            i++;
          } else {
            RssiReading += (char)LoRa.read();
          }
        }
        rxRSSI[counter-2] = RssiReading.toInt();
        RssiReading = "";
        if (counter == 1) Serial.println(String(phone));
      }
      else if (source == doneSubGnodeB){
        digitalWrite(led,LOW);
        txMode();
        break;
      }
      delayTime = millis();
    }
    timer = currentTime-delayTime;
    if (timer >= 1000){
      counter = 0;
      digitalWrite(led,LOW);
      if(dataRec){
        Serial.print("RSSI: Datasize = ");
        dataSize = sizeof(rxRSSI)/4;
        Serial.println(dataSize);
        for(int i=0;i<dataSize;i++){
          Serial.print(rxRSSI[i]);
          Serial.print(" ");
          if(i==19||i==39||i==59) Serial.println();
        }
        Serial.println("Data received waiting for go signal");
      }
      dataRec = 0;
    }
    if (currentTime >= 4294967200){
      digitalWrite(reset, LOW);
    }
  }
}

void txMode(){
  // send data
  delay(9000);
  Serial.println("Sending to gateway");
  digitalWrite(led,HIGH);
  LoRa.beginPacket();
  LoRa.write(subNode);
  LoRa.print(String(phone));
  Serial.println(String(phone));
  LoRa.endPacket();
  for (int i=0; i<dataSize; i++){
    sprintf (buf, "%03i", rxRSSI[i]);
    LoRa.beginPacket();
    LoRa.write(subNode);
    LoRa.print(buf);
    Serial.println(buf);
    LoRa.endPacket();
    delay(30);
  }
  doneTx();
}

void doneTx() {
  memset (phone, 0, sizeof(phone));
  memset (rxRSSI, 0, sizeof(rxRSSI));
  delay(1000);
  digitalWrite(led, LOW);
}
