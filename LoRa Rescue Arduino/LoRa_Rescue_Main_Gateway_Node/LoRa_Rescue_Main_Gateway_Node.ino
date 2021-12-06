// LoRa.h library reference
// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

// define LED and reset pins
#define led 6
#define reset 4

byte gatewayBroadcast = 0x11;   // address of gnode A
byte subGnodeB = 0xBB;      // address of gnode B
byte subGnodeC = 0xCC;      // address of gnode C
byte doneSubGnodeB = 0xAB;
char gway='C';
char phone[10];
char hopData[15];
char isPhone;
int fromHop = 0;
int counter = 0;
int timer = 0;
int source;
String phoneNum0,phoneNum1;
int rxRSSI,hopRSSI[60];
int dataSize = 60;
String RssiReading;
unsigned long currentTime;
unsigned long delayTime;

//float n = 3.2;
//float dro = 1.5;
//int roRSSI = -32;
float d[60];

////////////////////////////////////////////////////////////////////
void setup() {
  //Enables pins
  digitalWrite(reset, HIGH);
  pinMode(reset,OUTPUT);
  pinMode(led,OUTPUT);
  phoneNum0.reserve(65);
  phoneNum0.reserve(65);
  RssiReading.reserve(33);
  //Starts the UART
  Serial.begin(115200);
  
  memset (hopRSSI, 0, dataSize);
  rxRSSI = 0;
  phoneNum0 = "";
  phoneNum1 = "";
  fromHop = 3;

  //Starts LoRa and blinks when it fails to connect
  if (!LoRa.begin(868E6)) {
    while (1){
      delay(2000);
      digitalWrite(led,HIGH);
      delay(2000);
      digitalWrite(led,LOW);
    }
  }
  
  // Sets the LoRa prefences as well as call functions during LoRa activity
  // modem.onTxDone(onTxDone);
  LoRa.setTxPower(18.5,PA_OUTPUT_PA_BOOST_PIN);
  LoRa.setSignalBandwidth(125E3);
  LoRa.setSpreadingFactor(9);

  Serial.println("Main Gateway Started!");
  rxMode();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
void loop() {
  // Set time
  currentTime = millis();

  // Parse the packet
  // received a packet
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    
    source = LoRa.read();
    /////////////////////////////////////
    //////////     RX RSSI     //////////
    if (source == gatewayBroadcast) {
      digitalWrite(led,HIGH);
      counter++;

      // Reads packet and saves it to phoneNum string
      int i = 0;
      while (LoRa.available()) {
        isPhone = (char)LoRa.read();
        if (isDigit(isPhone)) phone[i] = isPhone;
        i++;
      }

      // RSSI of packet
      // Saves the RSSI at the rxRSSI array
      rxRSSI = LoRa.packetRssi();

      // Send Data to Serial
      Serial.print("A");
      for(int i=0;i<10;i++) Serial.print(phone[i]);
      Serial.print(" ");
      Serial.println(rxRSSI);
      memset (phone, 0, sizeof(phone));
    }
    /////////////////////////////////////
    
    //////////////////////////////////////
    //////////     HOP RSSI     //////////
    if (source == subGnodeB || source == subGnodeC){
      digitalWrite(led,HIGH);
      counter++;
      
      if(source == subGnodeB){
        gway = 'B';
      } else if (source == subGnodeC){
        gway = 'C';
      }

      int i = 0;
      while (LoRa.available()){
        hopData[i] = (char)LoRa.read();
        i++;
      }

      // Send Data to Serial
      Serial.print(gway);
      for(int i=0;i<15;i++) Serial.print(hopData[i]);
      Serial.println();
      memset (hopData, 0, sizeof(hopData));
    }
    //////////////////////////////////////
  }
  digitalWrite(led,LOW);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Sets LoRa to receive mode
void rxMode(){
  delay(1000);
  LoRa.receive();                       // set receive mode
}

void gnodeBdone(){
  digitalWrite(led,LOW);
  delay(17000);
  digitalWrite(led,HIGH);
  LoRa.beginPacket();
  LoRa.write(doneSubGnodeB);
  LoRa.endPacket();
  digitalWrite(led,LOW);
}

void resetArduino(){
  digitalWrite(reset, LOW);
}