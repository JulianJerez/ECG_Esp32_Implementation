#include <math.h>
#define eps 0.001
#define conv_size 2
#define conv_size_end 2
#define output_model_size 3
#define flatten_size 1026
#define input_model_size 1

//conv_size depends on the number of filters use
// in_size --> N° of col. Input[row][in_size]
// in_size --> N° of col. Input[row][in_size]
// output_size --> N° of col. Output[#][output_size]
// Output[row][output_size] when the layer is not Convolution
// Output[filters][output_size]


//****************************************************
//Pading of the Convolutional and Pooling layers: SAME
//****************************************************


//Internal Variable for the operations with Pading
float input_pad[2065][conv_size_end]={};

//General Output of the Flatten layer
float out_flatten[flatten_size]={};

//////// ##########################  Create Matriz   #########################
float** createMatrix_2D(float row, float col)
{
    int i = 0;
    float **matrix = NULL;
    matrix = (float**)malloc(sizeof(float*) * row);
    if(matrix == NULL)
        printf("Matrix2D malloc failed\n");
    for(i=0; i<row; i++)
    {
        matrix[i] = (float*)malloc(sizeof(float) * col);
        if(matrix[i] == NULL)  
            printf("Matrix2D malloc failed\n");
    }

    return matrix;
}


//////// ##########################  Clear Matriz   #########################
void Clear_matriz_max(float input[][conv_size_end],float col,float in_size){
for(int j=0;j<col;j++){
  for(int i=0;i<in_size;i++){
  input[i][j]=0;
  }
 }
}

void Clear_matriz_input(float input[][input_model_size],float col,float in_size){
for(int j=0;j<col;j++){
  for(int i=0;i<in_size;i++){
  input[i][j]=0;
  }
 }
}

void Clear_matriz(float **input,float col,float in_size){
for(int j=0;j<col;j++){
  for(int i=0;i<in_size;i++){
  input[i][j]=0;
  }
 }
}

void Clear_vector(float input[],float in_size)
{
  for(int i=0;i<in_size;i++)
  {
    input[i] = 0;
  }
}

//////// ##########################  Add Layer   #########################
void  Add(float **input1,float input2[][conv_size_end],float **output,float row,float col){

  for(int i=0;i<row;i++){
   for(int j=0;j<col;j++){
      output[i][j]=input1[i][j]+input2[i][j];
  }
 }
 //Serial.println(F("Add Good... "));
}

//////// ##########################  Dense Layer   #########################
void Dense(float input[],float output[],float out_size,float in_size,const float W[][output_model_size],const float b[]) {

  for (int j = 0; j <out_size; ++j) {
    for (int i = 0; i <in_size; ++i){ 
    output[j] += W[i][j] * input[i];
    }
    output[j] += b[j];
  }
  //Serial.println(F("Dense Good... "));
}

//////// ##########################  Relu Layer   #########################
void Relu(float **input, float col, float in_size)
{
  float a=0;
  for(int i=0;i<in_size;i++)
  {
    for(int j=0;j<col;j++)
    {
      if(input[i][j]>0)
      {input[i][j] += a;}
      else
      {input[i][j]=0;}
    }
  }
  //Serial.println(F("Relu Good... "));
}

//////// ##########################  Maxpooling 1D Layer   #########################
void Maxpooling(float **input,float output[][conv_size_end],float in_size,int col,float stride, float kernel_size,float *output_size){

 /////////////// Pading //////////////////////////////////
  float out_size,pad;
  float dim=0;
  float epsi=0;
  out_size=(in_size/stride);
  out_size=ceil(out_size);              // Dimension de la salida del Avg,Maxpooling...
  pad=((out_size-1)*stride-in_size+kernel_size)*0.5;
  dim=in_size+2*pad;                    // Dimension de la salida del pading
 //Serial.print("Pading: ");
 //Serial.println(pad,3);
  int a=0;
  int b=0;
  b=int(pad);
  for(int i=0;i<in_size;i++){
  for (int j=0;j<int(col);j++)
  {
    a=i+b;
    input_pad[a][j]=input[i][j];
   }
  }
///////////////////////////////////////////////////////////
 float maximo=-100000;
 int h=0,count=0;
 for(int j=0;j<col;j++){
   for(int i=0;i<dim-kernel_size+1;i=i+stride){
      for(int k=0;k<kernel_size;k++){
        h=i+k;
    
    ////////////////////////////
    if(h==0 && floor(pad)==1 && count==0)
    {maximo = maximo+epsi;}
    else if(h==dim-1 && ceil(pad)==1 && count==(out_size-1))
    {maximo = maximo+epsi;}
        else
    { if(input_pad[h][j]>maximo)
      {maximo=input_pad[h][j];}
      else{maximo = maximo+epsi;}
    }
      }
      output[count][j]=maximo;
      count++;
      maximo=-100000;;
   }   
   count=0;
 }
 *output_size=out_size;
  Clear_matriz_max(input_pad,conv_size_end,2065);
 //Serial.println(F("Max Good... "));
}

//////// ##########################  Averagepooling 1D Layer   #########################
void Avgpooling(float **input,float **output,float in_size,int col,float stride, float kernel_size,float *output_size){

 /////////////// Pading //////////////////////////////////
  float out_size,pad;
  float dim=0;
  out_size=(in_size/stride);
  out_size=ceil(out_size);              // Dimension de la salida del Avg,Maxpooling...
  pad=((out_size-1)*stride-in_size+kernel_size)*0.5;
  dim=in_size+2*pad;                    // Dimension de la salida del pading
  int a=0;
  int b=0;
  b=int(pad);
  //Serial.print("Pading: ");
  //Serial.println(pad,3);
  for(int i=0;i<in_size;i++){
  for (int j=0;j<int(col);j++)
  {
    a=i+b;
    input_pad[a][j]=input[i][j];
   }
  }
///////////////////////////////////////////////////////////
 float suma=0;
 int h=0,count=0;
 for(int j=0;j<col;j++){
   for(int i=0;i<dim-kernel_size+1;i=i+stride){
      for(int k=0;k<kernel_size;k++){
        h=i+k;
        suma+=input_pad[h][j]; 
      }
      if(count == 0)
      {output[count][j]=suma*(1/(kernel_size-floor(pad)));}
      else if (count == (out_size-1))
      {output[count][j]=suma*(1/(kernel_size-ceil(pad)));}
      else {output[count][j]=suma*(1/kernel_size);}
      count++;
      suma=0;
   }   
   count=0;
 }
 *output_size=out_size;
  Clear_matriz_max(input_pad,conv_size_end,2065);
 //Serial.println(F("Avg Good... "));
}

//////// ##########################  Batch Normalization Layer   #########################
void Batch(float **input, float **output,const float gama[],const float beta[],const float mean[],const float variance[],float in_size,float col){


for(int j=0;j<col;j++){
 for(int i=0;i<in_size;i++){
  
output[i][j]=((input[i][j]-mean[j])/(sqrt(variance[j]+eps)))*gama[j]+beta[j];
  }
 }
 //Serial.println(F("Batch Good... "));
}

//////// ##########################  Convolution 1D Layer   #########################

void Convolution_input(float input[][input_model_size],const float kernel[][input_model_size],float **output,const float bias[],float in_size,float col,int filters,float stride, float kernel_size,float *output_size){
  
  /////////////// Pading //////////////////////////////////
  float out_size,pad;
  int dim;
  out_size=(in_size/stride);
  out_size=ceil(out_size);              // Dimension de la salida del Avg,Maxpooling...
  pad=((out_size-1)*stride-in_size+kernel_size)*0.5;
  dim=in_size+2*pad;                    // Dimension de la salida del pading
 // float input_pad[dim][int(col)]={};
  int a=0;
  int b=0;
  b=int(pad);
  //Serial.print("Pading: ");
  //Serial.println(pad,3);
  for(int i=0;i<in_size;i++){
  for (int j=0;j<int(col);j++)
  {
    a=i+b;
    input_pad[a][j]=input[i][j];
   }
  }

///////////////// Convolucion //////////////////////////////////
 int p=0;
 int h=0,count=0;
 float epsi=0;

 for(int j=0;j<filters;j++){    
    for(int i=0;i<dim-kernel_size+1;i=i+stride){
     for(int k=0;k<kernel_size;k++){
      for(int t=0;t<col;t++){
        // h=i+k;
         p=k+j*kernel_size;
         output[count][j] += kernel[p][t]*input_pad[i+k][t];  
     } 
    }
   output[count][j]+=bias[j];
      
//   /////// Relu ///////////
//   if(output[count][j]>0)
//     {output[count][j] += epsi;}
//      else
//     {output[count][j] = 0;}  
//   ////////////////////////               
   count++; 
  }
  count = 0;
 }
 *output_size=out_size;
 Clear_matriz_max(input_pad,1,2065);
 //Serial.println(F("Conv Good... "));
}

void Convolution(float **input,const float kernel[][conv_size],float **output,const float bias[],float in_size,float col,int filters,float stride, float kernel_size,float *output_size){
  /////////////// Pading //////////////////////////////////
  float out_size,pad;
  int dim;
  out_size=(in_size/stride);
  out_size=ceil(out_size);              // Dimension de la salida del Avg,Maxpooling...
  pad=((out_size-1)*stride-in_size+kernel_size)*0.5;
  dim=in_size+2*pad;                    // Dimension de la salida del pading
  int a=0;
  int b=0;
  b=int(pad);
  //Serial.print(F("Pading: "));
  //Serial.println(pad,3);
  for(int i=0;i<in_size;i++){
  for (int j=0;j<int(col);j++)
  { 
    a=i+b;
    input_pad[a][j]=input[i][j];
   }
  }
 
///////////////////////////////////////////////////////////

/////////////// Convolucion //////////////////////////////////
 int p=0;
 int h=0,count=0;
 float epsi=0;
// float operation=0;
 for(int j=0;j<filters;j++){              
    for(int i=0;i<dim-kernel_size+1;i=i+stride){
     for(int k=0;k<kernel_size;k++){
      for(int t=0;t<col;t++){
         h=i+k;
         p=k+j*kernel_size;
         output[count][j] += kernel[p][t]*input_pad[h][t];
     }
    }
 output[count][j]+=bias[j];
//   /////// Relu ///////////
//   if(output[count][j]>0)
//      {output[count][j] += epsi;}
//      else
//      {output[count][j] = 0;}  
//   ////////////////////////                 
   count++; 
  }
  count = 0;
 }
 *output_size=out_size;
  Clear_matriz_max(input_pad,conv_size_end,2065);
 //Serial.println(F("Conv Good... "));
}

void Convolution32(float **input,const float kernel[][conv_size_end],float **output,const float bias[],float in_size,float col,int filters,float stride, float kernel_size,float *output_size){
  /////////////// Pading //////////////////////////////////
  float out_size,pad;
  int dim;
  out_size=(in_size/stride);
  out_size=ceil(out_size);              // Dimension de la salida del Avg,Maxpooling...
  pad=((out_size-1)*stride-in_size+kernel_size)*0.5;
  dim=in_size+2*pad;                    // Dimension de la salida del pading
  int a=0;
  int b=0;
  b=int(pad);
  //Serial.print(F("Pading: "));
  //Serial.println(pad,3);
  for(int i=0;i<in_size;i++){
  for (int j=0;j<int(col);j++)
  { 
    a=i+b;
    input_pad[a][j]=input[i][j];
   }
  }
 
///////////////////////////////////////////////////////////

/////////////// Convolucion //////////////////////////////////
 int p=0;
 int h=0,count=0;
 float epsi=0;
// float operation=0;
 for(int j=0;j<filters;j++){              
    for(int i=0;i<dim-kernel_size+1;i=i+stride){
     for(int k=0;k<kernel_size;k++){
      for(int t=0;t<col;t++){
         h=i+k;
         p=k+j*kernel_size;
         output[count][j] += kernel[p][t]*input_pad[h][t];
     }
    }
 output[count][j]+=bias[j];
//   /////// Relu ///////////
//   if(output[count][j]>0)
//      {output[count][j] += epsi;}
//      else
//      {output[count][j] = 0;}  
//   ////////////////////////                 
   count++; 
  }
  count = 0;
 }
 *output_size=out_size;
  Clear_matriz_max(input_pad,conv_size_end,529);
 //Serial.println(F("Conv Good... "));
}


//////// ##########################  Flatten Layer   #########################
void Flatten(float **input,float col, float in_size){
  int count=0;
    for(int j=0; j<in_size;j++){
      for(int i=0; i<col;i++)
        {
          out_flatten[count] = input[j][i];
          count += 1;
        }
      }
 //Serial.println(F("Flatten Good... "));
}

//////// ##########################  Lambda Layer   #########################
void Lambda(float input[][1],float output[][conv_size], float in_size, float col){
  for(int j=0;j<col;j++)
  {
    for(int i=0;i<in_size;i++)
    {
      output[i][j] = input[i][j];
    }
  }
}

//////// ########################## Softmax Layer   #########################
void Softmax(float z[output_model_size],float output[], int outputs){
  float suma = 0;
  float maximo = -1000;
  float t[outputs]={};

  for (int k=0; k<outputs;k++){
     if(z[k]>=maximo)
     {maximo=z[k];}
     else{maximo += 0;}
  }

//  //SOFTMAX NORMALIZED
//  for (int k=0; k<outputs;k++)
//  {
//      if(maximo>88)
//      {z[k] = (z[k]*88)/maximo;}
//      else{z[k] += 0;}
//      t[k] = exp(z[k]);
//	suma+=t[k];
//  }
//  
//  for (int i=0;i<outputs;i++)
//  {output[i]=t[i]/suma;}
  
  //TAKE THE MAXIMUM
  for(int i=0;i<outputs;i++)
    {
    if(z[i]==maximo)
    {output[i]=1;}
    else
    {output[i]=0;}
    }
    
  //Serial.println(F("Softmax Good... "));
}
