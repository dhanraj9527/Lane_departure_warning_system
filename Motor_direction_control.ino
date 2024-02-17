#include <ArduinoJson.h>
#include <SoftwareSerial.h>
#include <Servo.h>
String str = "";

#define rx 10
#define tx 11

DynamicJsonDocument doc(1024);
char c = 'S';
int b = 0;
Servo s;
int initial = 90;
int pos = 0;
void ardjson();
SoftwareSerial mySerial = SoftwareSerial(rx,tx);
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  mySerial.begin(9600);
  s.attach(9);
  pinMode(3, OUTPUT);
  pinMode(5, OUTPUT);
  s.write(initial);
  digitalWrite(3, 0);
}

void loop() {
  // put your main code here, to run repeatedly:
  ardjson();
  Serial.println(b);
  if (c == 'R' && b > 10) {
    digitalWrite(3, 50);
    digitalWrite(5, 50);
    for (pos = initial; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      Serial.println("in if");
      s.write(pos); // tell servo to go to position in variable 'pos'
      if (pos == 180) {
        initial = 180;
      }
      delay(15);                       // waits 15ms for the servo to reach the position
    }

  }
  else if (c == 'L' && b < -10) {
    digitalWrite(3, 50);
    digitalWrite(5, 50);
    for (pos = initial; pos >= 0; pos -= 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      Serial.println("in if");
      s.write(pos); // tell servo to go to position in variable 'pos'
      if (pos == 0) {
        initial = 0;
      }
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    //    s.write(0);
  }
  else {
    digitalWrite(3, 0);
    digitalWrite(5, 0);
  }
}

void ardjson()
{
  if (Serial.available() > 0) {
    str = Serial.readStringUntil('*');
    Serial.print(str);
  }
  else {
    Serial.println("Data not received");
  }
  DeserializationError error = deserializeJson(doc, str);
  b = doc["b"];
  if (doc["r"] == "R") {
    c = 'R';
  }
  else if (doc["r"] == "L") {
    c = 'L';
  }
  Serial.println(b);
}
