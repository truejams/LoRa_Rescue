#include <WiFi.h>
// https://github.com/me-no-dev/AsyncTCP
//#include <AsyncTCP.h>
// https://github.com/me-no-dev/ESPAsyncWebServer
#include <ESPAsyncWebServer.h>
#include <WiFiAP.h>
#include <WiFiClient.h>
#include "esp32-hal-cpu.h"

// Sets the Serial2 RX and TX pins
#define RXD2 16
#define TXD2 17

// Insert your own wifi network SSID and Password
const char *ssid = "LoRa Rescue Mobile Node";
const char *password = "12345678";

const int led = 2;
String phoneNumber = "0";
String phoneNumberTemp = "0";
String loraState = "0";
int serialComm = 1;
int dev = 0;
int delayMult = 0;

const char* PHON_INPUT = "value";
const char* TRAN_INPUT = "value";

// Instatiate the AsyncWebServer object on port 80
AsyncWebServer webServer(80);

///////////////////////////////////////////////
/////           Start of webpage          /////
///////////////////////////////////////////////

// Declare the webpage
// HTML comments look like this <! comment in between here >
const char htmlCode[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML>
<html id="bgColor">
  <style>
    html {
      font-family: 'Trebuchet MS', sans-serif; 
      display: inline-block; 
      text-align: center;
    }
    h1 {
      font-size: 2.9rem;
    }
    h2 {font-size: 2.1rem;}
    h3 {font-size: 1.9rem;}
    p {font-size: 1.7rem;}
    body {max-width: 400px; margin:0px auto; padding-bottom: 30px;}
    small {
      font-size: 12px;
    }
    input {
      text-align: center;
      height: 20px;
      border-radius: 12px;
      border-style: solid;
      border-width: 3px;
    }
    ::-webkit-input-placeholder {
      text-align: center;
    }
    .button {
      background-color: #33cccc;
      border: none;
      color: white;
      padding: 12px 12px;
      text-align: center;
      border-radius: 24px;
      text-decoration: none;
      display: inline-block;
      font-size: 14px;
      margin: 4px 4px;
      cursor: pointer;
      display: none;
      animation: blueCycle 3s linear infinite alternate;
    }
    .bu {
      font-weight: 900;
      font-size: 16px;
      display: none;
    }
    #numCheck{
      color: #8a3323;
      display: none;
    }
    #snackbar { 
      visibility: hidden;
      margin: auto;
      height: 70px;
      bottom: 120px;
      width: 120px;
      background-color: rgb(255, 255, 255);
      border-color: #33cccc;
      color: #000000;
      text-align: center;
      z-index: 2;
      font-size: 16px;
      padding: 4px;
      border-radius: 32px;
      border-style: solid;
      border-width: 3px;
    }
    #snackbar.show {
      visibility: visible;-
      animation: fadein 0.5s, fadeout 0.5s 19.5s;
    }
    @keyframes fadein { 
      from {bottom: 0; opacity: 0;}
      to {bottom: 15px; opacity: 1;}
    }
    @keyframes fadeout {
      from {bottom: 15px; opacity: 1;}
      to {bottom: 0; opacity: 0;}
    }
    @keyframes blueCycle {
      from {background-color:#33cccc;}
      to {background-color:#269b9b;}
    }
    @keyframes greyCycle {
      from {color:#000000;}
      to {color:#b9b9b9;} 
    }
  </style>
  
  <head>
    <title>LoRa Rescue</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
  
    <h1><span style="color: #33cccc;"><strong>LoRa Rescue</strong></span></h1>
    <h2><span id="phoneValue">#</span></h2>
    <p>Enter your phone number:</p>
    
    <div>
      <input id="phone" name="phone" onchange="updatePhone(this)" required="" type="tel" placeholder="place input here">
    </div>
    <div>
      <small>Format: ###-###-####</small>
    </div>
    <div id="numCheck"><br><br><strong>
      Error: Invalid Number</strong>
    </div>
    <br>
    <div><br>
      <button id="bu" class="button" onclick="buttClick()"><strong> Transmit LoRa </strong></button>
    </div>
    <br><br>
    <div id="snackbar">
      <br><strong>Transmitting</strong>
      <br><small>to gateway nodes</small>
    </div>
  
    <script>
      var phoneNumber = 0;
      var stopSpam = 0;
      var send = 1;

      function updatePhone(element) {
        phoneNumber = document.getElementById("phone").value;
        document.getElementById("phoneValue").innerHTML = phoneNumber;
        console.log(phoneNumber);
        var httpRequest = new XMLHttpRequest();
        httpRequest.open("GET", "/phone?value="+phoneNumber, true);
        httpRequest.send();
        showButton();
      }

      function showButton(){
        if(phoneNumber.length <= 9 || phoneNumber.length >= 15 || !phoneNumber.startsWith("9") && !phoneNumber.startsWith("0") && !phoneNumber.startsWith("+")) {
          document.getElementById("numCheck").style.display = 'inline-block';
          document.getElementById("bu").style.display = 'none';
        } else {
          document.getElementById("bu").style.display = 'inline-block';
          document.getElementById("numCheck").style.display = 'none';
        }
      }
      
      function buttClick(){
        var x = document.getElementById("snackbar");
        if(!stopSpam){
          if(send){
            x.className = "show";
            document.getElementById("phoneValue").style.animation = 'greyCycle 1s ease infinite alternate';
            document.getElementById("bu").style.backgroundColor = '#8a3323';
            document.getElementById("bu").style.animation = 'none';
            document.getElementById("bu").innerHTML = "<strong>     Stop Transmitting     </strong>";
            send = 0;
            var bPress = 1;
            var httpRequest = new XMLHttpRequest();
            httpRequest.open("GET", "/update?value="+bPress, true);
            httpRequest.send();
          }
          else if(!send){
            x.className = x.className.replace("show", "");
            document.getElementById("phoneValue").style.animation = 'none';
            document.getElementById("bu").style.backgroundColor = '#33cccc';
            document.getElementById("bu").style.animation = 'blueCycle 3s linear infinite alternate';
            document.getElementById("bu").innerHTML = "<strong>  Retransmit LoRa  </strong>";
            send = 1;
            var bPress = 0;
            var httpRequest = new XMLHttpRequest();
            httpRequest.open("GET", "/update?value="+bPress, true);
            httpRequest.send();
          }          
        }
      }
    </script>
  </body>
</html>
)rawliteral";

///////////////////////////////////////////////
/////            End of webpage           /////
///////////////////////////////////////////////

void setup(){
  setCpuFrequencyMhz(80); //Set CPU clock to 80MHz
  // Begin Serial Communications over USB
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1,  RXD2, TXD2);
  WiFi.softAP(ssid, password);
  pinMode(led, OUTPUT);
  
  // Print the IP Address of your device
  //Serial.println(WiFi.localIP());
  IPAddress myIP = WiFi.softAPIP();
  Serial.println(myIP);
  
  // Detail the route for root / web page
  webServer.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send_P(200, "text/html", htmlCode);
  });
  
  // Send a GET request to <ESP_IP>/phone?value=<inputMessage>
  webServer.on("/phone", HTTP_GET, [] (AsyncWebServerRequest *request) {
    String inputMessage;
    // GET input value on <ESP_IP>/phone?value=<inputMessage>
    if (request->hasParam(PHON_INPUT)) {
      inputMessage = request->getParam(PHON_INPUT)->value();
      phoneNumber = inputMessage;
      phoneNumber.trim();
      phoneNumberTemp = phoneNumber;
      phoneNumberTemp.remove(4,1);
      if(phoneNumberTemp == "1234"){
        phoneNumberTemp = phoneNumber;
        phoneNumberTemp.remove(0,4);
        if (phoneNumberTemp == "0") phoneNumberTemp = "1";
        developerMode(phoneNumberTemp.toInt()*100);
      }
      if(phoneNumberTemp == "4321"){
        phoneNumberTemp = phoneNumber;
        phoneNumberTemp.remove(0,4);
        if (phoneNumberTemp == "0") phoneNumberTemp = "1";
        developerMode(phoneNumberTemp.toInt()*1000);
      }
    }
    else {
      inputMessage = "Did not receive";
    }
    
    
    phoneNumber.replace("+", "");
    phoneNumber.replace("-", "");
    phoneNumber.replace(" ", "");
    if(phoneNumber.startsWith("63")) phoneNumber.remove(0,2);
    if(phoneNumber.startsWith("0")) phoneNumber.remove(0,1);
    Serial.println(phoneNumber);
    request->send(200, "text/plain", "OK");
  });

  // Send a GET request to <ESP_IP>/update?value=<LoRaState>
   webServer.on("/update", HTTP_GET, [] (AsyncWebServerRequest *request) {
    String inputMessage1;
    // GET input1 value on <ESP_IP>/update?value=<LoRaState>
    if (request->hasParam(TRAN_INPUT)) {
      inputMessage1 = request->getParam(TRAN_INPUT)->value();
      loraState = inputMessage1;
    }
    else {
      inputMessage1 = "Did not receive";
    }
    Serial.print("LoRa: ");
    Serial.println(inputMessage1);
    request->send(200, "text/plain", "OK");
  });
  
  // Start server (remembering its on port 80)
  webServer.begin();
}
  
void loop() {
  if(loraState == "1"){
    digitalWrite(led,HIGH);
    Serial2.println(phoneNumber);
    if (dev == 0) delay(3000);
    if (dev == 1) delay(delayMult);
  }
  else if (loraState == "0") {
    digitalWrite(led,LOW);
    Serial2.println(0);
    delay(100);
  }
}

void developerMode(int mult) {
  delayMult = mult;
  if (dev == 0){
    dev = 1;
    for (int i = 0; i<9; i++){
      digitalWrite(led,HIGH);
      delay(100);
      digitalWrite(led,LOW);
      delay(100);
    }
  } else {
    dev = 0;
    for (int i = 0; i<3; i++){
      digitalWrite(led,HIGH);
      delay(300);
      digitalWrite(led,LOW);
      delay(300);
    }
  }
}
