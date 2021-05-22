#include "knn.h"


double distance(Image *a, Image *b) {
  // TODO: Return correct distance
  double dis = 0;  
  
  for (int i = 0; i < (a->sx * a->sy); i++)  
  {  
    dis = dis + pow(a->data[i] - b->data[i],2);  
  }  
  dis = sqrt(dis);  
    
  return dis;
}


void insertion_sort(int* label, int* dis, int num) {  
  
  int value_dis;  
  int hole;  
  int value_label;  
  int i;  
   
  for(i = 1; i < num; i++) {   
    value_dis = dis[i];  
    value_label = label[i];  
    hole = i;  
  
    while (hole > 0 && dis[hole - 1] > value_dis) {  
      dis[hole] = dis[hole - 1];  
      label[hole] = label[hole - 1];  
      hole--;  
    }  
  
    if(hole != i) {  
      dis[hole] = value_dis;  
      label[hole] = value_label;  
    }  
  }   
}


int knn_predict(Dataset *data, Image *input, int K) {  
  //TODO: Replace this with predicted label (0-9)  
  
  int label[K + 1];  
  int dis[K + 1];  
  
  int i = 0;  
  
  while (i < K)  
  {  
    label[i] = data->labels[i];  
    dis[i] = distance(data->images + i, input);  
    i++;  
  }  
  
  insertion_sort(label, dis, K);  
  
  int last = i;  
  
  while (i < data->num_items)  
  {  
    label[last] = data->labels[i];  
    dis[last] = distance(data->images + i, input);  
  
    insertion_sort(label, dis, K + 1);  
    i++;  
  }  
    
  int max_app = -10;  
  int curr_app;  
  int value = -10;  
  int j;  
  
  i = 0;  
  
  while(i < K)  
  {   
    curr_app = 1;  
  
    j = i + 1;  
    while (j < K)  
    {   
      if (label[j] == label[i]){  
         curr_app++;  
      }  
      j++;  
    }  
  
    if (curr_app > max_app) {  
      max_app = curr_app;   
      value = label[i];  
    }  
    else if (max_app == curr_app){  
      if (label[i] <= value)  
      {  
        value = label[i];  
      }  
    }   
    i++;  
  }      
  return value;  
}


Dataset *load_dataset(const char *filename) {  
  FILE* file = fopen(filename,"r");  
  
  int image_size = 28;  
  
  if(file != NULL)  
  {  
    int num_elements;  
    if (fread(&num_elements,4,1,file) != 1)  
    {  
      fprintf(stderr, "Can not read number of elements.\n");  
      exit(1);  
    }  
    Dataset* new_data = (Dataset*) malloc(sizeof(Dataset));  
    if (new_data == NULL)  
    {  
      fprintf(stderr,"Do not have enough space.");  
      exit(1);  
    }  
  
    new_data->num_items = num_elements;  
    new_data->images = (Image*) malloc(sizeof(Image) * num_elements);  
    if (new_data->images == NULL)  
    {  
      fprintf(stderr,"Do not have enough space.");  
      exit(1);  
    }  
  
    new_data->labels = (unsigned char *) malloc(sizeof(unsigned char) * num_elements);  
    if (new_data->labels == NULL)  
    {  
      fprintf(stderr,"Do not have enough space.");  
      exit(1);  
    }  
  
    int i = 0;  
    while (i < num_elements)  
    {   
      if (fread(&new_data->labels[i], 1, 1, file) != 1)  
      {  
        fprintf(stderr, "Can not read label from data.\n");  
        exit(1);  
      }  
      
      new_data->images[i].sx = image_size;  
      new_data->images[i].sy = image_size;  
      new_data->images[i].data = (unsigned char *) malloc(sizeof(unsigned char) * image_size * image_size);  
      if (new_data->images[i].data == NULL)  
      {  
        fprintf(stderr,"Do not have enough space.");  
        exit(1);  
      }  
        
      if (fread(new_data->images[i].data, 1, image_size * image_size, file) != image_size * image_size)  
      {  
        fprintf(stderr, "Can not read data.\n");  
        exit(1);  
      }  
        
      i++;  
    }  
      
    fclose(file);  
    return new_data;  
  }  
  return NULL;  
}  


void free_dataset(Dataset *data) {  
  for (int i = 0; i < data->num_items; i++)   
  {  
    free(data->images[i].data);  
  }  
  free(data->images);  
  free(data->labels);  
  free(data);  
  return;  
}


void child_handler(Dataset *training, Dataset *testing, int K, 
                   int p_in, int p_out) {
  // TODO: Compute number of correct predictions from the range of data 
  //      provided by the parent, and write it to the parent through `p_out`.

  int start_idx;
  if (read(p_in, &start_idx, sizeof(int)) == -1) {
    fprintf(stderr, "Can not read from parent.\n");  
    exit(1);
  }

  int N;
  if (read(p_in, &N, sizeof(int)) == -1) {
    fprintf(stderr, "Can not read from parent.\n");  
    exit(1);
  }

  int correct = 0;
  int label;
  int i = start_idx;
  while (i < (start_idx + N))
  { 
    label = knn_predict(training, &testing->images[i], K);
    if(testing->labels[i] == label)
    {
      correct = correct + 1;
    }
    i++;
  }

  if (write(p_out, &correct, sizeof(int)) != sizeof(int)) {
    fprintf(stderr, "Can not write to parent.\n");  
    exit(1);
  }
  return;
}