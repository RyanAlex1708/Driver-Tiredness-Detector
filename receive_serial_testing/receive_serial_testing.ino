int LED = 13;
int SPEAKER = 2;
int data;
unsigned long startTime;

void setup() {
  Serial.begin(9600);
  pinMode(LED, OUTPUT);
  pinMode(SPEAKER, OUTPUT);
  digitalWrite(LED, LOW);
}

void loop() {
    if (Serial.available() > 0) {
        data = Serial.read();
        startTime = millis();
        while (millis() - startTime <= 5000) {
            digitalWrite(LED, HIGH);
            digitalWrite(SPEAKER, HIGH); // turn on speaker
            delay(200);
            digitalWrite(LED, LOW);
            digitalWrite(SPEAKER, LOW); // turn off speaker
            delay(200);
        }
    }
}