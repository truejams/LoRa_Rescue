// LoRa.h library reference
// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

// define LED and reset pins
#define led 6
#define reset 1

byte gatewayBroadcast = 0x11;     // address of this device
byte subNode = 0xCC;
byte doneSubGnodeB = 0xAC;
int dataRec = 0;
int counter = 0;
int timer = 0;
char isPhone;
char phone[10];
int rxRSSI;
byte source;
char buf [4];
int dataSize = 60;
unsigned long currentTime;
unsigned long delayTime;

////////////////////////////////////////////////////////////////////
void setup() {
  //Enables pins
  digitalWrite(reset, HIGH);
  pinMode(reset,OUTPUT);
  pinMode(led,OUTPUT);
  
  //Starts the UART
  Serial.begin(115200);
  delay(2000);
  Serial.println("Gateway Node");

  memset (phone, 0, sizeof(phone));
  rxRSSI = 0;
  
  //Starts LoRa and blinks when it fails to connect
  if (!LoRa.begin(868E6)) {
    Serial.println("Starting LoRa failed!");
    while (1){
      delay(2000);
      digitalWrite(led,HIGH);
      delay(2000);
      digitalWrite(led,LOW);
    }
  } else {
    Serial.println("LoRa Started");
  }
  
  // Sets the LoRa prefences as well as call functions during LoRa activity
  LoRa.setTxPower(18.5,PA_OUTPUT_PA_BOOST_PIN);
  LoRa.setSignalBandwidth(125E3);
  LoRa.setSpreadingFactor(9);

  rxMode();
}

////////////////////////////////////////////////////////////////////
void loop() {
  // Set time
  currentTime = millis();

  // Parse the packet
  if (LoRa.parsePacket()) {
    // received a packet
    digitalWrite(led,HIGH);
    source = LoRa.read();

    // RSSI of packet
    // Saves the RSSI at the rxRSSI array
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
      
      // Send Data to LoRa
      digitalWrite(led,LOW);
      delay(300);
      digitalWrite(led,HIGH);
      sprintf (buf, " %03i", rxRSSI);
      LoRa.beginPacket();
      LoRa.write(subNode);
      LoRa.print(String(phone));
      LoRa.print(buf);
      LoRa.endPacket();
      Serial.print(subNode);
      Serial.print(String(phone));
      Serial.println(buf);
      rxMode();
      memset (phone, 0, sizeof(phone));
    }
  }
  digitalWrite(led,LOW);

  
  // Looks to see if receiving is done and resets counter
//  timer = currentTime-delayTime;

  // Sends if data is received and 1000ms has passed
//  if (timer >= 500){
//    counter = 0;
//    digitalWrite(led,LOW);
//
//    // displays RSSI
//    if(dataRec){
//      Serial.print("RSSI: Datasize = ");
//      dataSize = sizeof(rxRSSI)/4;
//      Serial.println(dataSize);
//      for(int i=0;i<dataSize;i++){
//        Serial.print(rxRSSI[i]);
//        Serial.print(" ");
//        if(i==19||i==39||i==59) Serial.println();
//      }
//      digitalWrite(led,HIGH);
//      delay(6500);
//      txMode(); // Starting this line change for stuff
//    }
//    dataRec = 0;
//  }
  
  // resets arduino board once the runtime is near 50 days (internal clock overflows)
//  if (currentTime >= 4294967200){
//    digitalWrite(reset, LOW);
//  }
}

////////////////////////////////////////////////////////////////////
// Sets LoRa to receive mode
void rxMode(){
  LoRa.disableInvertIQ();               // normal mode
  LoRa.receive();                       // set receive mode
}

////////////////////////////////////////////////////////////////////
// This is to be modified for other gateways to hop the data to the main gateway.
// Further research must be conducted to how the LoRaWAN gateway receives data.
// Anyway, the sent data here has an inverted I Q which cannot be read by other gateway nodes.
// Note that the sent data contains 1 phone number and the RSSI array saved earlier
void txMode(){
  // send data
  delay(15000);
  Serial.println("Tx mode");
  LoRa.beginPacket();
  LoRa.write(subNode);
  String sendPhone = String(phone);
  LoRa.print(sendPhone);
  Serial.println(sendPhone);
  LoRa.endPacket();
  for (int i=0; i<dataSize; i++){
    sprintf (buf, "%03i", rxRSSI);
    LoRa.beginPacket();
    LoRa.write(subNode);
    Serial.println(buf);
    LoRa.print(buf);
    LoRa.endPacket();
    delay(30);
  }
  sendPhone = "";
  doneTx();
}

// This function is called once the transmitting of the phone number to the database is done
void doneTx() {
  memset (phone, 0, sizeof(phone));
//  memset (rxRSSI, 0, sizeof(rxRSSI));
  Serial.println("Tx Done");
  delay(1000);
  rxMode();
}
