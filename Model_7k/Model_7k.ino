#include "functions.h"
#include "model_parameters_7k.h"
#include "ESP32TimerInterrupt.h"
#include "ESP32_ISR_Timer.h"
#define clear_size 2049
#define clear_max 1025
#define clear_end 513
#define kernel_size 16
#define pool_size 2
#define out_size 3
#define stride 2
#define stride_conv 1
#define media_input 503.0602
#define varianza_input 109.8372

//DEFINICION DE  LOS OBJETOS PARA LAS INTERRUPCIONES
hw_timer_t * timer = NULL;
volatile int conteo_input=0;
volatile double toma_senal[clear_size]={};
const int ledPin = 13;
float output_max[clear_max][conv_size_end]={};
float senal_entrada[clear_size][1]={};

int conteo_interno;
int periodo, led_flag;
float senal[clear_size] = {};
float output_final[output_model_size]={};
float in_size,col, filters, in_size1;

/////DEFINICION DE LAS INTERRUPCIONES
void IRAM_ATTR lectura(){
  
  if(conteo_input < clear_size)
  {
    conteo_input = conteo_input;
  }
  else{
    conteo_input=0;
    for(int i=0;i<clear_size;i++)
    {
      toma_senal[i]=0;
    }
    }
  //Serial.println(conteo_input);
//  if((digitalRead(10) == 1)||(digitalRead(11) == 1)){
//    //Serial.println('!');//*
//    if(conteo_input==0)
//    {
//      toma_senal[conteo_input]=500;
//    }
//    else
//    {
//      toma_senal[conteo_input]=toma_senal[(conteo_input-1)];
//    }
//  }
//  else{
//    // send the value of analog input 0:
//    //Serial.println(analogRead(14));//*
//    toma_senal[conteo_input]= analogRead(14);
//  }
  toma_senal[conteo_input]=input[conteo_input][1];
  //Serial.println(toma_senal[conteo_input]);
  conteo_input = conteo_input+1;
}


void setup() {
Serial.begin(115200);
//pinMode(ledPin, OUTPUT); 
pinMode(10, INPUT); // Setup for leads off detection LO +
pinMode(11, INPUT); // Setup for leads off detection LO -
periodo = 500000;
led_flag = 0;

//Serial.println("INICIO DE LA RED");

//LA PRIMERA SEÑAL PARA LA INFERENCIA ES UNA SEÑAL DEL BANCO
for(int i=0; i<clear_size;i++){
  senal[i] = input[i][1];
}

//LLAMADO DE LAS INTERRUPCIONES
timer = timerBegin(0, 316, true);
timerAttachInterrupt(timer, &lectura, true);
timerAlarmWrite(timer, 1000, true);
timerAlarmEnable(timer);
}

void led()
{
  if(led_flag==0)
  {
  digitalWrite(ledPin, LOW);  
  led_flag=1;
  } 
  else{
  digitalWrite(ledPin, HIGH);
  led_flag=0;  
  }
}

void loop() {

//VARIABLES INTERMEDIAS GENERALES
float **output_1 = createMatrix_2D(clear_size,conv_size);
float **output_2 = createMatrix_2D(clear_size,conv_size);
float **output_1_32 = createMatrix_2D(clear_end,conv_size_end);
float **output_2_32 = createMatrix_2D(clear_end,conv_size_end);
in_size = clear_size;
col=1;
filters=2;
Clear_matriz_input(senal_entrada,input_model_size,clear_size);

////ASIGNACION DE LA ENTRADA PARA LA INFERENCIA 
for(int i=0;i<clear_size;i++)
{
  senal_entrada[i][1]=senal[i];
  //senal_entrada[i][1]=(senal[i]-media_input)/varianza_input;
  //Serial.println(senal_entrada[i][1]);
}
Clear_vector(senal,clear_size);
 

Serial.println("INICIA LA INFERENCIA");
digitalWrite(ledPin, LOW);
//////////////////////////////////////////////////////////////
/////////////////PARTE INICIAL///////////////////////////////
Clear_matriz(output_1,conv_size,clear_size);
Clear_matriz(output_2,conv_size,clear_size);
Clear_vector(output_final, output_model_size);
Clear_vector(out_flatten,flatten_size);


Convolution_input(senal_entrada,W1,output_2,b1,in_size,col,filters,stride_conv,kernel_size,&in_size);
col=filters;
Batch(output_2,output_1, gamma_1,beta_1,mean_1,variance_1,in_size,col);
Relu(output_1,col,in_size);

///////////////////// BLOQUE INICIAL ////////////////////////////
in_size1 = clear_size;
Maxpooling(output_1,output_max,in_size1,col,stride,pool_size,&in_size1);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W2,output_2,b2,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2,output_2, gamma_2,beta_2,mean_2,variance_2,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W3,output_1,b3,in_size,col,filters,stride,kernel_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Add(output_1,output_max,output_2,in_size,col);
Clear_matriz(output_1,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE INICIAL");
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////PRIMER BLOQUE///////////////////////////////////////////////////
Maxpooling(output_2,output_max,in_size1,col,stride,pool_size,&in_size1);

Batch(output_2,output_1, gamma_3,beta_3,mean_3,variance_3,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W4,output_2,b4,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2,output_2, gamma_4,beta_4,mean_4,variance_4,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W5,output_1,b5,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Avgpooling(output_1,output_2,in_size,col,stride,pool_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Add(output_2,output_max,output_1,in_size,col);
Clear_matriz(output_2,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 1");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////SEGUNDO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1,output_2, gamma_5,beta_5,mean_5,variance_5,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W6,output_1,b6,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1,output_1, gamma_6,beta_6,mean_6,variance_6,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W7,output_2,b7,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Avgpooling(output_2,output_1,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Add(output_1,output_max,output_2,in_size,col);

Clear_matriz(output_1,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 2");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TERCER BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_2,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_2,output_1, gamma_7,beta_7,mean_7,variance_7,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W8,output_2,b8,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2,output_2, gamma_8,beta_8,mean_8,variance_8,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W9,output_1,b9,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Avgpooling(output_1,output_2,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Add(output_2,output_max,output_1,in_size,col);

Clear_matriz(output_2,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 3");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////CUARTO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1,output_2, gamma_9,beta_9,mean_9,variance_9,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W10,output_1,b10,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1,output_1, gamma_10,beta_10,mean_10,variance_10,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W11,output_2,b11,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Avgpooling(output_2,output_1,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Add(output_1,output_max,output_2,in_size,col);

Clear_matriz(output_1,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 4");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////QUINTO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_2,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_2,output_1, gamma_11,beta_11,mean_11,variance_11,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W12,output_2,b12,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2,output_2, gamma_12,beta_12,mean_12,variance_12,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W13,output_1,b13,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Avgpooling(output_1,output_2,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Add(output_2,output_max,output_1,in_size,col);

Clear_matriz(output_2,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 5");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////SEXTO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1,output_2, gamma_13,beta_13,mean_13,variance_13,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W14,output_1,b14,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1,output_1, gamma_14,beta_14,mean_14,variance_14,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W15,output_2,b15,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Avgpooling(output_2,output_1,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Add(output_1,output_max,output_2,in_size,col);

Clear_matriz(output_1,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 6");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////SEPTIMO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_2,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_2,output_1, gamma_15,beta_15,mean_15,variance_15,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W16,output_2,b16,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2,output_2, gamma_16,beta_16,mean_16,variance_16,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W17,output_1,b17,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Avgpooling(output_1,output_2,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Add(output_2,output_max,output_1,in_size,col);

Clear_matriz(output_2,col,clear_size);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 7");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////OCTAVO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1,output_2, gamma_17,beta_17,mean_17,variance_17,in_size,col);
Relu(output_2,col,in_size);

Clear_matriz(output_1,col,clear_size);
Convolution(output_2,W18,output_1,b18,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1,output_1, gamma_18,beta_18,mean_18,variance_18,in_size,col);
Relu(output_1,col,in_size);

Clear_matriz(output_2,col,clear_size);
Convolution(output_1,W19,output_2,b19,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1,col,clear_size);
Avgpooling(output_2,output_1,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2,col,clear_size);
Add(output_1,output_max,output_2,in_size,col);

Clear_matriz(output_1,col,clear_size);
Clear_matriz_max(output_max,conv_size_end,clear_max);

//Serial.println("BLOQUE 8");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////NOVENO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_2,output_max,in_size1,col,stride,pool_size,&in_size1);

Batch(output_2,output_1, gamma_19,beta_19,mean_19,variance_19,in_size,col);
Relu(output_1,col,in_size);

filters=4;

////aqui van los float del segundo

Clear_matriz(output_2_32,conv_size_end,clear_end);

//En esta convolucion se da el cambio de filters
Convolution(output_1,W20,output_2_32,b20,in_size,col,filters,stride_conv,kernel_size,&in_size);

col=filters;

Batch(output_2_32,output_2_32, gamma_20,beta_20,mean_20,variance_20,in_size,col);
Relu(output_2_32,col,in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Convolution32(output_2_32,W21,output_1_32,b21,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Avgpooling(output_1_32,output_2_32,in_size,col,stride,pool_size,&in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Add(output_2_32,output_max,output_1_32,in_size,col);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 9");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DECIMO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1_32,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1_32,output_2_32, gamma_21,beta_21,mean_21,variance_21,in_size,col);
Relu(output_2_32,col,in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Convolution32(output_2_32,W22,output_1_32,b22,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1_32,output_1_32, gamma_22,beta_22,mean_22,variance_22,in_size,col);
Relu(output_1_32,col,in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Convolution32(output_1_32,W23,output_2_32,b23,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Avgpooling(output_2_32,output_1_32,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Add(output_1_32,output_max,output_2_32,in_size,col);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 10");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DECIMO PRIMER BLOQUE/////////////////////////////////////////////////
Maxpooling(output_2_32,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_2_32,output_1_32, gamma_23,beta_23,mean_23,variance_23,in_size,col);
Relu(output_1_32,col,in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Convolution32(output_1_32,W24,output_2_32,b24,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2_32,output_2_32, gamma_24,beta_24,mean_24,variance_24,in_size,col);
Relu(output_2_32,col,in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Convolution32(output_2_32,W25,output_1_32,b25,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Avgpooling(output_1_32,output_2_32,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Add(output_2_32,output_max,output_1_32,in_size,col);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 11");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DECIMO SEGUNDO BLOQUE////////////////////////////////////////////////////////
Maxpooling(output_1_32,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_1_32,output_2_32, gamma_25,beta_25,mean_25,variance_25,in_size,col);
Relu(output_2_32,col,in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Convolution32(output_2_32,W26,output_1_32,b26,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_1_32,output_1_32, gamma_26,beta_26,mean_26,variance_26,in_size,col);
Relu(output_1_32,col,in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Convolution32(output_1_32,W27,output_2_32,b27,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Avgpooling(output_2_32,output_1_32,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Add(output_1_32,output_max,output_2_32,in_size,col);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 12");
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DECIMO TERCER BLOQUE/////////////////////////////////////////////////
Maxpooling(output_2_32,output_max,in_size1,col,stride_conv,pool_size,&in_size1);

Batch(output_2_32,output_1_32, gamma_27,beta_27,mean_27,variance_27,in_size,col);
Relu(output_1_32,col,in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Convolution32(output_1_32,W28,output_2_32,b28,in_size,col,filters,stride_conv,kernel_size,&in_size);

Batch(output_2_32,output_2_32, gamma_28,beta_28,mean_28,variance_28,in_size,col);
Relu(output_2_32,col,in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Convolution32(output_2_32,W29,output_1_32,b29,in_size,col,filters,stride_conv,kernel_size,&in_size);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Avgpooling(output_1_32,output_2_32,in_size,col,stride_conv,pool_size,&in_size);

Clear_matriz(output_1_32,conv_size_end,clear_end);
Add(output_2_32,output_max,output_1_32,in_size,col);

Clear_matriz(output_2_32,conv_size_end,clear_end);
Clear_matriz_max(output_max,col,clear_max);

//Serial.println("BLOQUE 11");

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////PARTE FINAL///////////////////////////////////////////////////////////////////
Batch(output_1_32,output_1_32, gamma_29,beta_29,mean_29,variance_29,in_size,col);
Relu(output_1_32,col,in_size); 

Flatten(output_1_32,col,in_size);
in_size=flatten_size;
 
Dense(out_flatten,output_final,out_size,in_size,Wd,bd);

/////IMPRIMIR LA SALIDA ANTES DE LA SOFTMAX
Serial.println("Salida:");
for(int i=0;i<out_size;i++){
  Serial.println(output_final[i],3);
}

////CALCULO DE LA SOFTMAX
Softmax(output_final,output_final,out_size);

//IMPRIMIR LA SALIDA DE LA SOFTMAX
Serial.println("Salida Softmax:");
for(int i=0;i<out_size;i++){
  Serial.println(output_final[i],3);
}

//TIEMPO DE DURACION DE LA INFERENCIA
unsigned long end_inference = millis();
Serial.println("Tiempo fin inferencia: ");
Serial.println(end_inference);
////digitalWrite(ledPin, LOW); 
Serial.println("FIN DE LA INFERENCIA");

//ACA ESTABA INICIALMENTE LA ASIGNACION DEL PERIODO


////DEMORA MIENTRAS SE TERMINA LA TOMA DE DATOS 
conteo_interno=0;
while(conteo_interno<clear_size){
  conteo_interno=conteo_input;
}

/////ASIGNACION DEL PERIODO DEL LED
//if((output_final[0]==1) && (output_final[1]==0) && (output_final[2]==0) )
//{periodo = 1000000;}
//else if((output_final[0]==0) && (output_final[1]==1) && (output_final[2]==0))
//{periodo = 100000;}
//else
//{
//  if((output_final[0]==0) && (output_final[1]==0) && (output_final[2]==1))
//  {periodo=1000;}
//  else
//  {periodo=3000000;}
//}


/////ASIGNACIÓN DE LA ENTRADA NUEVA DE LA INFERENCIA
Clear_vector(senal,clear_size);
noInterrupts();
for(int i=0;i<clear_size;i++)
{
  senal[i] = toma_senal[i];
}
interrupts();

/////TIEMPO DE DURACION TOTAL 
unsigned long end_red = millis();
Serial.println("FINALIZA LA RED");
Serial.println(end_red);

}
