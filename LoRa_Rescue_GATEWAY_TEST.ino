// LoRa.h library reference
// https://github.com/sandeepmistry/arduino-LoRa/blob/master/API.md

#include <SPI.h>
#include <LoRa.h>

char phone[11] = "9976500649";
int dat;

////////////////////////////////////////////////////////////////////
void setup() {
  Serial.begin(115200);
  delay(5000);
}

////////////////////////////////////////////////////////////////////
void loop() {
  Serial.print("A1");
  for(int i=0;i<10;i++) Serial.print(phone[i]);
  Serial.println();
  for(int i=0;i<60;i++){
    Serial.print("A2");
    dat = -32 - random(0,4);
    Serial.println(dat);
  }
  Serial.println("A3");
  delay(5000);
  Serial.print("B1");
  for(int i=0;i<10;i++) Serial.print(phone[i]);
  Serial.println();
  for(int i=0;i<60;i++){
    Serial.print("B2");
    dat = -32 - random(0,4);
    Serial.println(dat);
  }
  Serial.println("B3");
  delay(5000);
  Serial.print("C1");
  for(int i=0;i<10;i++) Serial.print(phone[i]);
  Serial.println();
  for(int i=0;i<60;i++){
    Serial.print("C2");
    dat = -32 - random(0,4);
    Serial.println(dat);
  }
  Serial.println("C3");
  delay(20000);
}
