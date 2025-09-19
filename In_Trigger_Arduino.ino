// title: "In-Trigger Arduino"
// date: "09/19/2025"
// author: "Babur Erdem"
// update date: "09/19/2025"
// Protocol @115200: "N<pin>\n" -> digitalWrite(pin, HIGH), "F<pin>\n" -> LOW.
// Example: N13\n turns the onboard LED on.

#include <Arduino.h>

const uint8_t ALLOWED_PINS[] = {2,3,4,5,6,7,8,9,10,11,12,13}; // edit to match your wiring
const size_t NP = sizeof(ALLOWED_PINS)/sizeof(ALLOWED_PINS[0]);

char cmd = 0;       // 'N' or 'F'
int pinNum = -1;    // parsed pin number

bool isAllowed(int p){
  for(size_t i=0;i<NP;i++) if(ALLOWED_PINS[i]==p) return true;
  return false;
}

void execCmd(char c, int p){
  if(!isAllowed(p)) return;
  pinMode(p, OUTPUT);       // safe if called multiple times
  digitalWrite(p, (c=='N') ? HIGH : LOW);
}

void flushIfReady(){
  if(cmd && pinNum >= 0){
    execCmd(cmd, pinNum);
    cmd = 0; pinNum = -1;
  }
}

void setup(){
  Serial.begin(115200);
  // Optional: initialize all allowed pins LOW
  for(size_t i=0;i<NP;i++){ pinMode(ALLOWED_PINS[i], OUTPUT); digitalWrite(ALLOWED_PINS[i], LOW); }
}

void loop(){
  while(Serial.available()){
    char c = Serial.read();
    if(c=='N' || c=='F'){ cmd = c; pinNum = -1; }                 // start new command
    else if(c>='0' && c<='9'){                                   // build pin number
      if(pinNum < 0) pinNum = 0;
      pinNum = pinNum*10 + (c - '0');
    } else if(c=='\n' || c=='\r' || c==' ' || c=='\t'){          // delimiter -> execute
      flushIfReady();
    } else {
      // ignore other chars
    }
  }
  // no delay; fully non-blocking
}
