
//compile with -D, e.g
//
// gcc -fopenmp -o manbrot mandlebrot.c -DDYNAMIC
//
//to get the version that uses dynamic scheduling

#include <omp.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float timediff(struct timespec t1, struct timespec t2)
{  if (t1.tv_nsec > t2.tv_nsec) {
     t2.tv_sec -= 1;
     t2.tv_nsec += 1000000000;
   }
   return t2.tv_sec-t1.tv_sec + 0.000000001 * (t2.tv_nsec-t1.tv_nsec);
}


#ifdef RC
//finds chunk among 0,....,n-1 to assign to thread number me among nth
//threads
void findmyrange(int n, int nth, int me, int *myrange)
{ int chunksize = n / nth;
  myrange[0] = me * chunksize;
  if (me < nth-1) myrange[1] = (me+1) * chunksize -1;
  else myrange[1] = n - 1;
}

//returns a random permutation of 0..n-1
int *rpermute(int n) {
  int *a = (int *) (int *) malloc(n*sizeof(int));
  int k;
  for (k=0; k < n; k++) a[k] = k;
  for (k=n-1; k > 0; k--) {
    int j = rand() % (k+1);
    int temp = a[j];
    a[j] = a[k];
    a[k] = temp;
  }
  return a;
}
#endif

#define MAXITERS 1000

//globals
int count = 0;
int nptsside;
float side2;
float side4;

int inset(double complex c) {
  int iters;
  float rl,im;
  double complex z = c;
  for (iters = 0; iters < MAXITERS; iters++) {
    z = z*z +c;
    rl = creal(z);
    im = cimag(z);
    if (rl*rl + im*im > 4) return 0;

  }
  return 1;
}


int *scram;

void dowork()
{
  #ifdef RC
  #pragma omp parallel reduction(+:count)
  #else
  #pragma omp parallel
  #endif
  {
    #pragma omp single
    { printf("Using %d threads\n", omp_get_num_threads()); }

    int x,y; float xv, yv;
    
    double complex z;
    #ifdef STATIC
    #pragma omp for reduction(+:count) schedule(static)
    #elif defined DYNAMIC
    #pragma omp for reduction(+:count) schedule(dynamic)
    #elif defined GUIDED
    #pragma omp for reduction(+:count) schedule(guided)
    #endif

    #ifdef RC
    int myrange[2];
    int me = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int i;
    findmyrange(nptsside, nth, me, myrange);
    for (i=myrange[0]; i <= myrange[1]; i++) {
      x = scram[i];
    #else

    for (x=0; x< nptsside; x++) {
    #endif
       for (y=0; y < nptsside; y++) {
         xv = (x - side2) / side4;
	 yv = (y - side2) / side4;
	 z = xv + yv*I;
	 if (inset(z)) {
	   count++;
	 }
       }
    }
  }
}

int main(int argc, char **argv)
{
  nptsside = atoi(argv[1]);
  side2 = nptsside / 2.0;
  side4 = nptsside / 4.0;

  struct timespec bgn,nd;
  clock_gettime(CLOCK_REALTIME, &bgn);

  #ifdef RC
  scram = rpermute(nptsside);
  printf("Random chunk RC is defined\n");
  #endif
  
  dowork();

  //implied barrier
  printf("%d\n", count);
  clock_gettime(CLOCK_REALTIME, &nd);
  printf("%f\n", timediff(bgn,nd));

}
