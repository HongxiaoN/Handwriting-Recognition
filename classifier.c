#include "knn.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets
//
// Example of running validation (K = 3, 8 processes):
//    ./classifier 3 datasets/training_data.bin datasets/testing_data.bin 8


/**
 * main() takes in 4 command line arguments:
 *   - K:  The K value for kNN
 *   - training_data: A binary file containing training image / label data
 *   - testing_data: A binary file containing testing image / label data
 *   - num_procs: The number of processes to be used in validation
 * 
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the testing set among childred by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should gets N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers don't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test 
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 */


// Helper func to send error message and close data.
void error(Dataset* training_data, Dataset* testing_data)
{
  fprintf(stderr,"Can not operate with pipe.");
  free_dataset(training_data);
  free_dataset(testing_data);
}


int main(int argc, char *argv[]) {

  if (argc != 5) {  
      fprintf(stderr, "The arguments are not equal to 4.\n");  
      exit(1);  
  }  
  int K = atoi(argv[1]);  
  char* file_training = argv[2];  
  char* file_testing = argv[3]; 
  int num_procs = atoi(argv[4]);
  
  Dataset* training_data = load_dataset(file_training);  
  Dataset* testing_data = load_dataset(file_testing);

  // calculate how many image will be process in each child.
  int N = ceil((double) testing_data->num_items / (double) num_procs);
  int start_idx = 0;

  // pipe the number of process of child read end and write end.
  int pipe_fd[num_procs][2];

  int i = 0;
  while (i < num_procs)
  { 
    // check is pipe available
    if (pipe(pipe_fd[i]) == -1) {
      fprintf(stderr, "Can not pipe.\n");  
      exit(1);
    } 

    pid_t p;
    if ((p = fork()) < 0) {
      fprintf(stderr,"Can not fork.");
      exit(1);
    }

    else if (p == 0) {
      int before = 0;
      while (before < i)
      { 
        if (close(pipe_fd[before][0]) == -1) {
          error(training_data,testing_data);
          exit(1);
        }
        before++;
      }
      child_handler(training_data, testing_data, K, pipe_fd[i][0], pipe_fd[i][1]);

      // close pipe read end and write end when we finish calculation.
      if (close(pipe_fd[i][0]) == -1) {
        error(training_data,testing_data);
        exit(1);
      }
      if (close(pipe_fd[i][1]) == -1) {
        error(training_data,testing_data);
        exit(1);
      }
      free_dataset(training_data);
      free_dataset(testing_data);
      exit(0);
    }
    else {
      
      int check = 0;
      
      if(start_idx < testing_data->num_items){

        if (write(pipe_fd[i][1], &start_idx, sizeof(int)) != sizeof(int)) {
          error(training_data,testing_data);
          exit(1);
        }
        if (i == num_procs - 1)
          N = testing_data->num_items - (i * N);

        if (write(pipe_fd[i][1], &N, sizeof(int)) != sizeof(int)) {
          error(training_data,testing_data);
          exit(1);
        }
      }
      else {
        if (write(pipe_fd[i][1], &check, sizeof(int)) != sizeof(int)) {
          error(training_data,testing_data);
          exit(1);
        }
        if (write(pipe_fd[i][1], &check, sizeof(int)) != sizeof(int)) {
          error(training_data,testing_data);
          exit(1);
        }
      }
      if (close(pipe_fd[i][1]) == -1) {
        error(training_data,testing_data);
        exit(1);
      }

      start_idx = start_idx + N;
    }
    i++;
  }

  int total = 0;

  int status;
  while (wait(&status) > 0);

  int correct;
  for (int i = 0; i < num_procs; i++) {

    if (read(pipe_fd[i][0], &correct, sizeof(int)) == -1) {
      error(training_data,testing_data);
      exit(1);
    }

    if (close(pipe_fd[i][0]) == -1) {
      error(training_data,testing_data);
      exit(1);
    }
    total = total + correct;
  }

  printf("%d\n", total);

  free_dataset(training_data);
  free_dataset(testing_data);
  return 0;
}
