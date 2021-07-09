// LoRa.h library reference
// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

// define LED and reset pins
#define led 6
#define reset 4

byte gatewayBroadcast = 0xAA;   // address of gnode A
byte subGnodeB = 0xBB;      // address of gnode B
byte subGnodeC = 0xCC;      // address of gnode C
byte doneSubGnodeB = 0xAB;
char gway='C';
char phone[10];
char isPhone;
int fromHop = 0;
int counter = 0;
int timer = 0;
int source;
String phoneNum0,phoneNum1;
int rxRSSI[60],hopRSSI[60];
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
  memset (rxRSSI, 0, dataSize);
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

////////////////////////////////////////////////////////////////////
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
    if (source == gatewayBroadcast && counter <= dataSize) {
      fromHop = 0;
      digitalWrite(led,HIGH);
      counter++;

      // Reads packet and saves it to phoneNum string
      int i = 0;
      while (LoRa.available()) {
        if(counter <= 1){
          isPhone = (char)LoRa.read();
          if (isDigit(isPhone)) phone[i] = isPhone;
          i++;
        } else {
          LoRa.read();
        }
      }

      // RSSI of packet
      // Saves the RSSI at the rxRSSI array
      rxRSSI[counter-1] = LoRa.packetRssi();

      // Display Number
      if(counter == 1){
        Serial.print("A1");
        for(int i=0;i<10;i++) Serial.print(phone[i]);
        Serial.println();
        memset (phone, 0, sizeof(phone));
      } 
    }
    /////////////////////////////////////
    
    //////////////////////////////////////
    //////////     HOP RSSI     //////////
    if (source == subGnodeB || source == subGnodeC && counter <= dataSize){
      fromHop = 1;
      counter++;
      
      digitalWrite(led,HIGH);
      if(source == subGnodeB){
        gway = 'B';
      } else if (source == subGnodeC){
        gway = 'C';
      }

      int i = 0;
      while (LoRa.available()){
        if(counter == 1){
          isPhone = (char)LoRa.read();
          if (isDigit(isPhone)) phone[i] = isPhone; 
          i++;
        } else {
          RssiReading += (char)LoRa.read();
        }
      }

      hopRSSI[counter-2] = RssiReading.toInt();
      RssiReading = "";
      if(counter == 1){
        Serial.print(gway);
        Serial.print("1");
        for(int i=0;i<10;i++) Serial.print(phone[i]);
        Serial.println();
        memset (phone, 0, sizeof(phone));
      }
    }
    //////////////////////////////////////
    
    delayTime = millis();
  }

  // Looks to see if receiving is done and resets counter
  timer = abs(currentTime - delayTime);
  
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  // This runs when a data is received and 500ms has passed.
  if (timer >= 500){
    counter = 0;
    // sends data to other gateway nodes if Rx is done
    // resets phoneNum after
    if (fromHop == 1){
      for(int i=0;i<dataSize;i++){
        Serial.print(gway);
        Serial.print("2");
        Serial.println(hopRSSI[i]);
      }
      Serial.print(gway);
      Serial.println("3");
      if(gway == 'B'){
        gnodeBdone();
      }
      memset (hopRSSI, 0, dataSize);
      memset (d, 0, dataSize);
      digitalWrite(led,LOW);
    } else if (fromHop == 0) {
      for(int i=0;i<dataSize;i++){
        Serial.print("A2");
        // RSSI To distance
        //d[i] = pow(10,((roRSSI-rxRSSI[i])/(10*n)))*dro;
        Serial.println(rxRSSI[i]);
      }
      Serial.println("A3");
      digitalWrite(led,LOW);
      memset (rxRSSI, 0, dataSize);
    }
    fromHop = 3;
  }
  
  // resets arduino board once the runtime is near 50 days (internal clock overflows)
  if (currentTime >= 4294967200){
    resetArduino();
  }
}

////////////////////////////////////////////////////////////////////

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
