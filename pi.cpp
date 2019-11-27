 
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include<bits/stdc++.h>
using namespace std;

const double PI = 3.1415926535897932;
const long STEP_NUM = 1070596096;
const double STEP_LENGTH = 1.0 / 1070596096;

int calculate_pi(int t)
{
  struct timeval startTime;
  gettimeofday(&startTime, NULL);

  double sum = 0.0;
  double pi, x;

  printf("\nStart calculating...\n");
  // computational steps
  #pragma omp parallel for reduction(+:sum) private(x) num_threads(t)
  for(int i = 0;i < STEP_NUM; i++)
  {
    x = (i + 0.5) * STEP_LENGTH;
    sum += 1.0 / (1.0 + x * x);
  }
  pi = STEP_LENGTH * sum * 4;

  struct timeval endTime;
  gettimeofday(&endTime, NULL);
  printf("PI = %.16lf with error %.16lf\nTime elapsed : %lf seconds.\n\n", pi, fabs(pi - PI), (endTime.tv_sec - startTime.tv_sec) + ((double)(endTime.tv_usec - startTime.tv_usec) / 10E6 ));
  assert(fabs(PI - pi) <= 0.001);
  return 0;
}

int main(){
  cout<<"SHOWING RESULTS FOR DIFFERENT NUMBER OF THREADS USING OPENMP: \n\n";
  cout<<"Threads: 1 (SEQUENCE ALGO)\n";
  calculate_pi(1);
  cout<<"\n\nThreads: 2 (PARALLEL ALGO)\n";
  calculate_pi(2);
  cout<<"\n\nThreads: 3 (PARALLEL ALGO)\n";
  calculate_pi(3);
  cout<<"\n\nThreads: 4 (PARALLEL ALGO)\n";
  calculate_pi(4);
  cout<<"\n\nThreads: 5 (PARALLEL ALGO)\n";
  calculate_pi(5);
  
}